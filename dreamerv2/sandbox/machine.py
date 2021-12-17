from loguru import logger as lgr
import re
from einops import rearrange
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common

from sandbox import attention, slot_attention

class EnsembleRSSM(common.Module):

  def __init__(
      self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
      act='elu', norm='none', std_act='softplus', min_std=0.1, dynamics='default', update='default', embed_dim=16, num_slots=1):
    super().__init__()
    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = get_act(act)
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    self.num_slots = num_slots
    self._dynamics_type = dynamics
    self._update_type = update
    self._embed_dim = embed_dim

    if self._dynamics_type == 'default':
      self.dynamics = DefaultDynamics(self._deter, self._hidden, self._act, self._norm)
    elif self._dynamics_type == 'cross_attention':
      self.dynamics = CrossAttentionDynamics(self._deter, self._hidden, self._act, self._norm)  # TODO: later manually set the number of slots for the specific episode
    elif self._dynamics_type == 'separate_embedding':
      self.dynamics = SeparateEmbeddingDynamics(self._deter, self._hidden, self._act, self._norm)
    elif self._dynamics_type == 'slim_cross_attention':
      self.dynamics = SlimCrossAttentionDynamics(self._deter, self._hidden, self._act, self._norm, self.num_slots)  # TODO: later manually set the number of slots for the specific episode
    else:
      raise NotImplementedError

    if self._update_type == 'default':
      self.update = DefaultUpdate(self._hidden, self._act, self._norm)
    elif self._update_type == 'slim_attention':
      self.update = SlimAttentionUpdate(self._deter, self._act, self._norm, self._embed_dim, self.num_slots)  # TODO: later manually set the number of slots for the specific episode
      # NOTE that I am treating self._deter = self._hidden
    elif self._update_type == 'slot_attention':
      self.update = SlotAttentionUpdate(self._deter, self._act, self._norm, self._embed_dim, self.num_slots)  # TODO: later manually set the number of slots for the specific episode
      # NOTE that I am treating self._deter = self._hidden
    else:
      raise NotImplementedError


  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self.dynamics._cell.get_initial_state(None, batch_size, dtype)  # initialized to zero
          )
      if self._update_type == 'slot_attention':
        slots = tf.cast(self.update.reset(batch_size), dtype)
        state['deter'] = state['deter'] + slots.reshape((batch_size, -1))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self.dynamics._cell.get_initial_state(None, batch_size, dtype)
          )
    return state

  @tf.function
  def observe(self, embed, action, is_first, state=None):
    """
      embed: (16, 50, 1536)
      action: (16, 50, 6)
      state: 
        logit: (B, S, V)
        stoch: (B, S, V)
        deter: (B, D)
    """
    swap = lambda x: rearrange(x, 'x y ... -> y x ...')
    if state is None:
      state = self.initial(tf.shape(action)[0])
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (swap(action), swap(embed), swap(is_first)), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    """
    t: seed steps
    T: total length
    H: prediction horizon. H = T-t

    action        (H, B, A)
    state logit   (B, S, V)
    state stoch   (B, S, V)
    state deter   (B, D)
    prior logit   (H, B, S, V)
    prior stoch   (H, B, S, V)
    prior deter   (H, B, D)
    """
    swap = lambda x: rearrange(x, 'x y ... -> y x ...')
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    # during sampling, we had cast stoch to f32, now we cast it back to f16
    stoch = self._cast(state['stoch'])
    if self._discrete:
      stoch = rearrange(stoch, '... s v -> ... (s v)')
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, ensemble=False):
    if ensemble:
      state = self._suff_stats_ensemble(state['deter'])
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
    # if is_first.any():
    prev_state, prev_action = tf.nest.map_structure(
        lambda x: tf.einsum(
            'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
        (prev_state, prev_action))
    prior = self.img_step(prev_state, prev_action, sample)
    ###########################################################
    # replace this with slot attention
    x = self.update(prior['deter'], embed)
    ###########################################################
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      prev_stoch = rearrange(prev_stoch, '... s v -> ... (s v)')
    ###########################################################
    # replace with this transformer
    x, deter = self.dynamics(prev_state['deter'], prev_stoch, prev_action)
    ###########################################################
    stats = self._suff_stats_ensemble(x)
    index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_ensemble(self, inp):
    # bs = list(inp.shape[:-1])
    # inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
      x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
      x = self._act(x)
      stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
    stats = {
        k: tf.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    # stats = {
    #     k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
    #     for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x):
    # import ipdb; ipdb.set_trace(context=20)
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = rearrange(x, '... (s v) -> ... s v', v=self._discrete)
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value

  def register_num_slots(self, num_slots):
    self.dynamics.register_num_slots(num_slots)
    self.update.register_num_slots(num_slots)


#############################################################
# Encoder
#############################################################

class Encoder(common.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
    self.shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    lgr.info('Encoder CNN inputs:', list(self.cnn_keys))
    lgr.info('Encoder MLP inputs:', list(self.mlp_keys))
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers

  @tf.function
  def __call__(self, data):
    """
      (B, T, 64, 64, 3) --> (B, T, 1536)
    """
    key, shape = list(self.shapes.items())[0]
    batch_dims = data[key].shape[:-len(shape)]
    data = {
        k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_keys:
      outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
    if self.mlp_keys:
      outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
    output = tf.concat(outputs, -1)
    return output.reshape(batch_dims + output.shape[1:])

  def _cnn(self, data):
    """
      (B*T, 64, 64, 3) --> (B*T, F, F, 384)
      384 = 2**3 * 48

      debug: 
      (B*T, 64, 64, 3)
      (B*T, 31, 31, 4)
      (B*T, 14, 14, 8)
      (B*T, 6, 6, 16)
      (B*T, 2, 2, 32)
      (B*T, 2*2*32) = (B*T, 128)
    """
    x = tf.concat(list(data.values()), -1)  # (B*T, H, W, C)
    x = x.astype(prec.global_policy().compute_dtype)
    for i, kernel in enumerate(self._cnn_kernels):
      depth = 2 ** i * self._cnn_depth
      x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
      x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x.reshape(tuple(x.shape[:-3]) + (-1,))

  def _mlp(self, data):
    x = tf.concat(list(data.values()), -1)
    x = x.astype(prec.global_policy().compute_dtype)
    for i, width in enumerate(self._mlp_layers):
      x = self.get(f'dense{i}', tfkl.Dense, width)(x)
      x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x

"""
TODO:
  - make resolution, outdim configurable
  - take out reshaping for slots
"""
class SlotEncoder(Encoder):
  def __init__(self, shapes, encoder_type, outdim, **kwargs):
    super().__init__(shapes, **kwargs)
    if encoder_type == 'slimslot':
      self.encoder = slot_attention.SlimSlotAttentionEncoder(
      resolution=(24, 24), outdim=outdim)  # hardcoded for now. Later will be agnostic to resolution and outdim will be larger
    elif encoder_type == 'slimmerslot':
      self.encoder = slot_attention.DebugSlotAttentionEncoder(
      resolution=(24, 24), outdim=outdim)  # hardcoded for now. Later will be agnostic to resolution and outdim will be larger
    elif encoder_type == 'slot':
      self.encoder = slot_attention.SlotAttentionEncoder(
      resolution=(64, 64), outdim=outdim)  # hardcoded for now. Later will be agnostic to resolution and outdim will be larger
    else:
      raise NotImplementedError

  def _cnn(self, data):
    """
    """
    x = tf.concat(list(data.values()), -1)  # (B*T, H, W, C)
    x = x.astype(prec.global_policy().compute_dtype)
    x = self.encoder(x)
    # hacky reshape just for testing
    x = x.reshape(tuple(x.shape[:-2]) + (-1,))  # TODO: this should actually be given by config
    return x


#############################################################
# Decoder
#############################################################


class Decoder(common.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
    self._shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    lgr.info('Decoder CNN outputs:', list(self.cnn_keys))
    lgr.info('Decoder MLP outputs:', list(self.mlp_keys))
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers

  def __call__(self, features):
    features = tf.cast(features, prec.global_policy().compute_dtype)
    outputs = {}
    if self.cnn_keys:
      outputs.update(self._cnn(features))
    if self.mlp_keys:
      outputs.update(self._mlp(features))
    return outputs

  def _cnn(self, features):
    """
      (16, 10, deter + num_tokens * stoch_size)
      (16, 10, hiddim) --> the discrete latents select codebook vectors and sum them
      (160, 1, 1, hiddim)  
      --> start with a 1x1, and then you end up distributing that across space.

      0 (B, 5, 5, 16)
      1 (B, 13, 13, 8)
      2 (B, 30, 30, 4)
      3 (B, 64, 64, 3)
    """
    channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
    ConvT = tfkl.Conv2DTranspose
    x = self.get('convin', tfkl.Dense, 32 * self._cnn_depth)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._cnn_depth])
    for i, kernel in enumerate(self._cnn_kernels):
      depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
      act, norm = self._act, self._norm
      if i == len(self._cnn_kernels) - 1:
        depth, act, norm = sum(channels.values()), tf.identity, 'none'
      x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
      x = self.get(f'convnorm{i}', NormLayer, norm)(x)
      x = act(x)
    x = x.reshape(features.shape[:-1] + x.shape[1:])  # (B, T, H, W, C)
    means = tf.split(x, list(channels.values()), -1)  # [(B, T, H, W, C)]
    dists = {
        key: tfd.Independent(tfd.Normal(mean, 1), 3)
        for (key, shape), mean in zip(channels.items(), means)}
    return dists

  def _mlp(self, features):
    shapes = {k: self._shapes[k] for k in self.mlp_keys}
    x = features
    for i, width in enumerate(self._mlp_layers):
      x = self.get(f'dense{i}', tfkl.Dense, width)(x)
      x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    dists = {}
    for key, shape in shapes.items():
      dists[key] = self.get(f'dense_{key}', DistLayer, shape)(x)
    return dists

"""
TODO:
  - make the output shape of the decoder configurable
  - take out reshaping for slots
"""
class SlotDecoder(Decoder):
  def __init__(self, shapes, indim, decoder_type, **kwargs):
    super().__init__(shapes, **kwargs)
    if decoder_type == 'slot':
      self.decoder = slot_attention.SlotAttentionDecoder(indim, (64, 64))  # hardcoded to (64, 64) with indim 112
    elif decoder_type == 'slimmerslot':
      self.decoder = slot_attention.DebugSlotAttentionDecoder6464(indim)  # hardcoded to (64, 64) with indim 112
    else:
      raise NotImplementedError

  def _cnn(self, features):
    """
      features: (B, T, D)
    """
    channels = {k: self._shapes[k][-1] for k in self.cnn_keys}

    # hacky reshape for now
    batch_size, seq_length = features.shape[:2]
    slot_dim = features.shape[-1]  # TODO: this should actually be given by config
    slots = features.reshape((batch_size*seq_length, -1, slot_dim))  # TODO: this should actually be given by config

    x = self.decoder(slots)

    # Undo combination of slot and batch dimension; split alpha masks.
    recons, masks = slot_attention.unstack_and_split(x, batch_size=batch_size*seq_length)
    # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    masks = tf.nn.softmax(masks, axis=1)
    recon_combined = tf.reduce_sum(recons * masks, axis=1)  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].

    x = recon_combined.reshape((batch_size, seq_length) + recon_combined.shape[-3:])
    #############################################################
    means = tf.split(x, list(channels.values()), -1)  # [(B, T, H, W, C)]
    dists = {
        key: tfd.Independent(tfd.Normal(mean, 1), 3)
        for (key, shape), mean in zip(channels.items(), means)}
    return dists


#############################################################
# Layers
#############################################################

class MLP(common.Module):

  def __init__(self, shape, layers, units, act='elu', norm='none', **out):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._norm = norm
    self._act = get_act(act)
    self._out = out

  def __call__(self, features):
    x = tf.cast(features, prec.global_policy().compute_dtype)
    x = x.reshape([-1, x.shape[-1]])
    for index in range(self._layers):
      x = self.get(f'dense{index}', tfkl.Dense, self._units)(x)
      x = self.get(f'norm{index}', NormLayer, self._norm)(x)
      x = self._act(x)
    x = x.reshape(features.shape[:-1] + [x.shape[-1]])
    return self.get('out', DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = get_act(act)
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  @tf.function
  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class DistLayer(common.Module):

  def __init__(
      self, shape, dist='mse', min_std=0.1, init_std=0.0):
    self._shape = shape
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std

  def __call__(self, inputs):
    out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
    out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
    out = tf.cast(out, tf.float32)
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
      std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
      std = tf.cast(std, tf.float32)
    if self._dist == 'mse':
      dist = tfd.Normal(out, 1.0)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * tf.tanh(out / 5)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, common.TanhBijector())
      dist = tfd.Independent(dist, len(self._shape))
      return common.SampleDist(dist)
    if self._dist == 'trunc_normal':
      std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
      return tfd.Independent(dist, 1)
    if self._dist == 'onehot':
      return common.OneHotDist(out)
    raise NotImplementedError(self._dist)


class NormLayer(common.Module):

  def __init__(self, name):
    if name == 'none':
      self._layer = None
    elif name == 'layer':
      self._layer = tfkl.LayerNormalization()
    else:
      raise NotImplementedError(name)

  def __call__(self, features):
    if not self._layer:
      return features
    return self._layer(features)


def get_act(name):
  if name == 'none':
    return tf.identity
  if name == 'mish':
    return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
  elif hasattr(tf.nn, name):
    return getattr(tf.nn, name)
  elif hasattr(tf, name):
    return getattr(tf, name)
  else:
    raise NotImplementedError(name)


#############################################################
# Dynamics
#############################################################
class DefaultDynamics(common.Module):
  def __init__(self, deter, hidden, act, norm):
    self._deter = deter
    self._hidden = hidden
    self._act = act
    self._norm = norm

    self._cell = GRUCell(self._deter, norm=True)

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots

  def __call__(self, prev_deter, prev_stoch, prev_action):
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', NormLayer, self._norm)(x)  # why do they normalize after the linear?
    x = self._act(x)
    x, deter = self._cell(x, [prev_deter])
    deter = deter[0]  # Keras wraps the state in a list.
    return deter, deter

"""
TODO
- make sure gradients backpropped into initial state
- make sure module recognizes the list of cross attention modules
"""
class CrossAttentionDynamics(common.Module):
  def __init__(self, deter, hidden, act, norm):
    self._deter = deter
    self._hidden = hidden
    self._act = act
    self._norm = norm

    # just to get the initial state for now
    self._cell = GRUCell(self._deter, norm=True)

    self.n_heads = 1  # TODO: make this a parameter in config.
    self.n_layers = 1
    self.cross_attention = [attention.CrossAttentionBlock(self._deter, self.n_heads, rate=0) for i in range(self.n_layers)]

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots

  def __call__(self, prev_deter, prev_stoch, prev_action):
    stoch_embed = self.get('stoch_embed', tfkl.Dense, self._hidden)(prev_stoch)
    act_embed =  self.get('act_embed', tfkl.Dense, self._hidden)(prev_action)
    context = tf.stack([stoch_embed, act_embed], 1)  # (B, K, H)

    batch_size = prev_deter.shape[0]
    query = prev_deter.reshape((batch_size, self.num_slots, self._deter//self.num_slots))  # (B, K, D)

    for cab in self.cross_attention:
      query = cab(query, context)  # (B, K, D)
      query = self.get('postnorm', NormLayer, self._norm)(query)

    deter = query.reshape((-1, self._deter))
    return deter, deter

# ok great, so this works.
class SlimCrossAttentionDynamics(common.Module):
  def __init__(self, deter, hidden, act, norm, num_slots):
    self._deter = deter
    self._hidden = hidden
    self._act = act
    self._norm = norm

    self.num_slots = num_slots

    self._cell = GRUCell(self._deter, norm=True)
    self.context_attention = attention.ContextAttention(self._deter//self.num_slots, num_heads=1)

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots

  def __call__(self, prev_deter, prev_stoch, prev_action):
    # lgr.info(prev_stoch.shape)  # (B, stoch*discrete)

    # need to reshape this to (B, num_slots, stoch/num_slots)

    # make this works such that a complete group of categories gets siphoned into the num_slots
    # assert False



    stoch_embed = self.get('stoch_embed', tfkl.Dense, self._hidden)(prev_stoch)




    act_embed =  self.get('act_embed', tfkl.Dense, self._hidden)(prev_action)
    context = tf.stack([stoch_embed, act_embed], 1)  # (B, K, H)

    batch_size = prev_deter.shape[0]
    query = prev_deter.reshape((batch_size, self.num_slots, self._deter//self.num_slots))  # (B, K, D)
    # lgr.info('SlimCrossAttentionDynamics query', query.shape)
    # lgr.info('SlimCrossAttentionDynamics context', context.shape)

    out = self.context_attention(query, context, mask=None)

    # lgr.info('SlimCrossAttentionDynamics out', out.shape)


    x = out.reshape((-1, self._deter))

    # lgr.info('SlimCrossAttentionDynamics x', x.shape)

    x = self.get('img_in_norm', NormLayer, self._norm)(x)  # why do they normalize after the linear?
    x = self._act(x)
    x, deter = self._cell(x, [prev_deter])
    deter = deter[0]  # Keras wraps the state in a list.
    return deter, deter


class SeparateEmbeddingDynamics(common.Module):
  def __init__(self, deter, hidden, act, norm):
    self._deter = deter
    self._hidden = hidden
    self._act = act
    self._norm = norm

    # just to get the initial state for now
    self._cell = GRUCell(self._deter, norm=True)

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots

  def __call__(self, prev_deter, prev_stoch, prev_action):
    stoch_embed = self.get('stoch_embed', tfkl.Dense, self._hidden)(prev_stoch)
    act_embed =  self.get('act_embed', tfkl.Dense, self._hidden)(prev_action)
    x = tf.concat([stoch_embed, act_embed], -1)  # (B, K, H)

    x = self.get('img_in', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', NormLayer, self._norm)(x)  # why do they normalize after the linear?
    x = self._act(x)
    x, deter = self._cell(x, [prev_deter])
    deter = deter[0]  # Keras wraps the state in a list.
    return deter, deter


#############################################################
# Update
#############################################################

class DefaultUpdate(common.Module):
  def __init__(self, hidden, act, norm):
    self._hidden = hidden
    self._act = act
    self._norm = norm

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots

  def __call__(self, deter, embed):
    x = tf.concat([deter, embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
    x = self.get('obs_out_norm', NormLayer, self._norm)(x)  # why do they normalize after the linear?
    x = self._act(x)
    return x


class SlimAttentionUpdate(common.Module):
  def __init__(self, deter, act, norm, embed_dim, num_slots):
    self._deter = deter
    self._act = act
    self._norm = norm
    self._embed_dim = embed_dim

    self.num_slots = num_slots

    self._cell = GRUCell(self._deter, norm=True)
    self.context_attention = attention.ContextAttention(self._deter//self.num_slots, num_heads=1)

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots

  def __call__(self, deter, embed):
    batch_size = deter.shape[0]

    context = embed.reshape((batch_size, -1, self._embed_dim))  # TODO
    query = deter.reshape((batch_size, self.num_slots, self._deter//self.num_slots))  # (B, K, D)

    # lgr.info('SlimAttentionUpdate context', context.shape)
    # lgr.info('SlimAttentionUpdate query', query.shape)

    out = self.context_attention(query, context, mask=None)

    # lgr.info('SlimAttentionUpdate out', out.shape)


    x = out.reshape((batch_size, self._deter))

    # lgr.info('SlimAttentionUpdate x', x.shape)


    x = self.get('obs_out_norm', NormLayer, self._norm)(x)  # why do they normalize after the linear?
    x = self._act(x)
    return x

class SlotAttentionUpdate(common.Module):
  def __init__(self, deter, act, norm, embed_dim, num_slots):
    self._deter = deter
    self._act = act
    self._norm = norm
    self._embed_dim = embed_dim

    self.num_slots = num_slots

    self.slot_attention = slot_attention.SlotAttention(
      num_iterations=3, 
      slot_size=self._deter,#//self.num_slots, 
      mlp_hidden_size=self._deter,
      learn_initial_dist=False)#//self.num_slots)
    # TODO: this needs to divide by self.num_slots
    # THIS ONLY WORKS BECAUSE WE ASSUME self.num_slots == 1!

    self.slot_attention.register_num_slots(num_slots)

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots
  #   self.slot_attention.register_num_slots(num_slots)

  def reset(self, batch_size):
    slots =  self.slot_attention.reset(batch_size)
    return slots

  def __call__(self, deter, embed):
    batch_size = deter.shape[0]

    context = embed.reshape((batch_size, -1, self._embed_dim))  # TODO
    query = deter.reshape((batch_size, self.num_slots, self._deter//self.num_slots))  # (B, K, D)

    # lgr.info('SlotAttentionUpdate context', context.shape)
    # lgr.info('SlotAttentionUpdate query', query.shape)


    updated_slots = self.slot_attention(query, context)

    # lgr.info('SlotAttentionUpdate updated_slots', updated_slots.shape)




    updated_slots = updated_slots.reshape((batch_size, -1))

    # lgr.info('SlotAttentionUpdate updated_slots reshaped', updated_slots.shape)



    return updated_slots

