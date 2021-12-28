from loguru import logger as lgr
import re
import einops as eo
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common

from sandbox import attention, slot_attention

# import pathlib
# import sys
# sys.path.append(str(pathlib.Path(__file__).parent / 'sandbox' / 'tf_slate'))
from sandbox.tf_slate import transformer, slot_attn, dvae
from sandbox import machine
from sandbox.machine import MLP, GRUCell, DistLayer, NormLayer, get_act, Encoder, Decoder


########################################################################
## utils
########################################################################
def unflatten(x, num_slots):
    return eo.rearrange(x, '... (k d) -> ... k d', k=num_slots)

def flatten(x):
    return eo.rearrange(x, '... k d -> ... (k d)')

def split_at_n(text, delimiter, n):
  groups = text.split(delimiter)
  return delimiter.join(groups[:n]), delimiter.join(groups[n:])

########################################################################
## Tensor utils
########################################################################

class SlotEnsembleRSSM(machine.EnsembleRSSM):

  def __init__(
      self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
      act='elu', norm='none', std_act='softplus', min_std=0.1, dynamics_type='default', update_type='default', initial_type='default', embed_dim=16, num_slots=1, resolution=(16,16)):
    # super().__init__()
    common.Module.__init__(self)
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
    self._dynamics_type = dynamics_type
    self._update_type = update_type
    self._embed_dim = embed_dim
    self._initial_type = initial_type
    self._resolution = resolution

    if self._dynamics_type == 'default':
      self.dynamics = DefaultDynamics(self._deter, self._hidden, self._act, self._norm)
    elif self._dynamics_type == 'cross_attention':
      self.dynamics = CrossAttentionDynamics(self._deter, self._hidden, self._act, self._norm)  # TODO: later manually set the number of slots for the specific episode
    elif self._dynamics_type == 'separate_embedding':
      self.dynamics = SeparateEmbeddingDynamics(self._deter, self._hidden, self._act, self._norm)
    elif self._dynamics_type == 'slim_cross_attention':
      self.dynamics = SlimCrossAttentionDynamics(self._deter, self._hidden, self._act, self._norm, self.num_slots)  # TODO: later manually set the number of slots for the specific episode
    elif self._dynamics_type == 'cross':
      self.dynamics = CrossDynamics(self._deter, self._hidden, self._act, self._norm)
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
    elif self._update_type == 'cross':
      self.update = CrossUpdate(self._hidden, self._act, self._norm)
    elif self._update_type == 'slot':
      self.update = SlotUpdate(self._hidden, self._act, self._norm)
    else:
      raise NotImplementedError

    if self._initial_type == 'iid':
      self.slots_mu = tf.Variable(tf.initializers.GlorotUniform()(shape=[self._deter]))
      self.slots_log_sigma = tf.Variable(tf.initializers.GlorotUniform()(shape=[self._deter]))
    elif self._initial_type == 'fixed':
      # like position encoding
      self.initial_deter = tf.Variable(tf.random.truncated_normal((self.num_slots, self._deter)))
    elif self._initial_type != 'default':
      raise NotImplementedError

    # # assert self.num_slots == 1
    # assert False


  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self.dynamics._cell.get_initial_state(None, batch_size, dtype)  # initialized to zero
          )
      if self._initial_type == 'fixed':
        broadcast = lambda x: eo.repeat(x, 'b ... -> b k ...', k=self.num_slots)
        state = tf.nest.map_structure(broadcast, state)
        state['deter'] = state['deter'] + tf.cast(self.initial_deter, dtype)
        if self.num_slots > 1:
          state['attns'] = tf.zeros([batch_size, np.prod(self._resolution), self.num_slots], dtype=dtype)
      elif self._initial_type == 'iid':
        deter = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal([batch_size, self.num_slots, self._deter])
        broadcast = lambda x: eo.repeat(x, 'b ... -> b k ...', k=self.num_slots)
        state = tf.nest.map_structure(broadcast, state)
        state['deter'] = state['deter'] + tf.cast(deter, dtype)
        if self.num_slots > 1:
          state['attns'] = tf.zeros([batch_size, np.prod(self._resolution), self.num_slots], dtype=dtype)
      elif self._initial_type != 'default':
        raise NotImplementedError
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self.dynamics._cell.get_initial_state(None, batch_size, dtype)
          )
    return state

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
    """
      prev_state:
        logit   (B, S, V)
        stoch   (B, S, V)
        deter   (B, D)
      action: (B, A)
      embed: (B, X)
      is_first: (B)
    """
    # # if is_first[idx] then we reset prev_state[idx] and zero prev_action[idx]
    # zero_first = lambda x: tf.einsum('b,b...->b...', 1.0 - is_first.astype(x.dtype), x)
    # zero_not_first = lambda x: tf.einsum('b,b...->b...', is_first.astype(x.dtype), x)

    # prev_action = zero_first(prev_action)
    # prev_state = tf.nest.map_structure(
    #   lambda x, y: zero_first(x) + zero_not_first(y).astype(x.dtype),
    #   prev_state, self.initial(tf.shape(prev_action)[0])
    #   )

    # prior = self.img_step(prev_state, prev_action, sample)
    # ###########################################################
    # # replace this with slot attention
    # if self.num_slots > 1:
    #   x, attns = self.update(prior['deter'], embed, return_attns=True)
    # else:
    #   x = self.update(prior['deter'], embed)
    # ###########################################################
    # stats = self._suff_stats_layer('obs_dist', x)
    # dist = self.get_dist(stats)
    # stoch = dist.sample() if sample else dist.mode()
    # post = {'stoch': stoch, 'deter': prior['deter'], **stats}

    post, prior = machine.EnsembleRSSM.obs_step(self, prev_state, prev_action, embed, is_first, sample)

    if self.num_slots > 1:
      post['attns'] = attns  # (B, H*W, K)

    try:
      tf.debugging.check_numerics(prior['stoch'], 'prior_stoch')
      tf.debugging.check_numerics(prior['logit'], 'prior_logit')
      tf.debugging.check_numerics(post['stoch'], 'post_stoch')
      tf.debugging.check_numerics(post['logit'], 'post_logit')
      tf.debugging.check_numerics(post['deter'], 'deter')
    except Exception as e:
      lgr.debug(e)
      import ipdb; ipdb.set_trace(context=20)


    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    # prev_stoch = self._cast(prev_state['stoch'])
    # prev_action = self._cast(prev_action)
    # if self._discrete:
    #   prev_stoch = eo.rearrange(prev_stoch, '... s v -> ... (s v)')
    # ###########################################################
    # # replace with this transformer
    # x, deter = self.dynamics(prev_state['deter'], prev_stoch, prev_action)
    # ###########################################################
    # # choose one from the ensemble to generate a dist
    # stats = self._suff_stats_ensemble(x)
    # index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    # stats = {k: v[index] for k, v in stats.items()}
    # dist = self.get_dist(stats)
    # stoch = dist.sample() if sample else dist.mode()
    # prior = {'stoch': stoch, 'deter': deter, **stats}

    prior = machine.EnsembleRSSM.img_step(self, prev_state, prev_action, sample)


    if self.num_slots > 1:
      prior['attns'] = self._cast(tf.zeros([prev_action.shape[0], 256, self.num_slots]))
    return prior

  def register_num_slots(self, num_slots):
    self.dynamics.register_num_slots(num_slots)
    self.update.register_num_slots(num_slots)


#############################################################
# Encoder
#############################################################

# class Encoder(common.Module):

#   def __init__(
#       self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
#       cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
#     self.shapes = shapes
#     self.cnn_keys = [
#         k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
#     self.mlp_keys = [
#         k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
#     lgr.info('Encoder CNN inputs:', list(self.cnn_keys))
#     lgr.info('Encoder MLP inputs:', list(self.mlp_keys))
#     self._act = get_act(act)
#     self._norm = norm
#     self._cnn_depth = cnn_depth
#     self._cnn_kernels = cnn_kernels
#     self._mlp_layers = mlp_layers

#   @tf.function
#   def __call__(self, data):
#     """
#       (B, T, 64, 64, 3) --> (B, T, 1536)
#     """
#     key, shape = list(self.shapes.items())[0]
#     batch_dims = data[key].shape[:-len(shape)]
#     data = {
#         k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
#         for k, v in data.items()}
#     outputs = []
#     if self.cnn_keys:
#       outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
#     if self.mlp_keys:
#       outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
#     output = tf.concat(outputs, -1)
#     return output.reshape(batch_dims + output.shape[1:])

#   def _cnn(self, data):
#     """
#       (B*T, 64, 64, 3) --> (B*T, F, F, 384)
#       384 = 2**3 * 48

#       debug: 
#       (B*T, 64, 64, 3)
#       (B*T, 31, 31, 4)
#       (B*T, 14, 14, 8)
#       (B*T, 6, 6, 16)
#       (B*T, 2, 2, 32)
#       (B*T, 2*2*32) = (B*T, 128)

#       actual:
#       (B*T, 64, 64, 3)
#       (B*T, 31, 31, 48)
#       (B*T, 14, 14, 96)
#       (B*T, 6, 6, 192)
#       (B*T, 2, 2, 384) = (B*T, 1536)

#       what I'd want

#       (B*T, 64, 64, 3)
#       (B*T, 31, 31, 48)
#       (B*T, 14, 14, 96)
#       position encoding
#       (B*T, 14, 14, 96)
#       (B*T, 14, 14, 96)

#     """
#     x = tf.concat(list(data.values()), -1)  # (B*T, H, W, C)
#     x = x.astype(prec.global_policy().compute_dtype)
#     for i, kernel in enumerate(self._cnn_kernels):
#       depth = 2 ** i * self._cnn_depth
#       x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
#       x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
#       x = self._act(x)
#     return eo.rearrange(x, '... h w c -> ... (h w c)')

#   def _mlp(self, data):
#     x = tf.concat(list(data.values()), -1)
#     x = x.astype(prec.global_policy().compute_dtype)
#     for i, width in enumerate(self._mlp_layers):
#       x = self.get(f'dense{i}', tfkl.Dense, width)(x)
#       x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
#       x = self._act(x)
#     return x

"""
TODO:
  - make resolution, outdim configurable
  - take out reshaping for slots
"""
class PreviousSlotEncoder(Encoder):
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


class GridEncoder(Encoder):
  def __init__(self, shapes, encoder_type, pos_encode_type, outdim, resolution, **kwargs):
    super().__init__(shapes, **kwargs)
    if encoder_type == 'grid_g':
      self.encoder = dvae.GenericEncoder(in_channels=3, out_channels=outdim)
    elif encoder_type == 'grid_dvweak':
      self.encoder = dvae.dVAEShallowWeakEncoder(in_channels=3, out_channels=outdim)
      # TODO: you need to add position embedding to this! 
    elif encoder_type == 'grid_dvstrong':
      self.encoder = dvae.dVAEStrongEncoder(in_channels=3, out_channels=outdim)
      # TODO: you need to add position embedding to this! 
    elif encoder_type == 'grid_sa':
      pass
    elif encoder_type == 'grid_saslim':
      pass
    elif encoder_type == 'grid_sadebug':
      pass
    else:
      raise NotImplementedError

    self.resolution = resolution
    if pos_encode_type == 'slate':
      self.position_encoding = transformer.GridPositionalEncoding(
        resolution=self.resolution, dim=outdim)
    elif pos_encode_type == 'coordconv':
      self.position_encoding = transformer.CoordConvPositionalEncoding(
        resolution=self.resolution, dim=outdim)
    elif pos_encode_type == 'sinusoid':
      pass
    elif pos_encode_type == 'none':
      self.position_encoding = lambda x: x
    else:
      raise NotImplementedError

    self.token_mlp = tf.keras.Sequential([
        tfkl.Dense(outdim, kernel_initializer='he_uniform'),
        tfkl.ReLU(),
        tfkl.Dense(outdim, kernel_initializer='he_uniform')])


    # then there is the tokenwise MLP. But note that that tokenwise MLP is not used for the target in slate

  def _cnn(self, data):
    """
    """
    x = tf.concat(list(data.values()), -1)  # (B*T, H, W, C)
    x = x.astype(prec.global_policy().compute_dtype)
    x = self.encoder(x)  # (B*T, H, W, D)
    x = self.token_mlp(self.position_encoding(x))
    return x


#############################################################
# Decoder
#############################################################


# class Decoder(common.Module):

#   def __init__(
#       self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
#       cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
#     self._shapes = shapes
#     self.cnn_keys = [
#         k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
#     self.mlp_keys = [
#         k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
#     lgr.info('Decoder CNN outputs:', list(self.cnn_keys))
#     lgr.info('Decoder MLP outputs:', list(self.mlp_keys))
#     self._act = get_act(act)
#     self._norm = norm
#     self._cnn_depth = cnn_depth
#     self._cnn_kernels = cnn_kernels
#     self._mlp_layers = mlp_layers

#   def __call__(self, features):
#     features = tf.cast(features, prec.global_policy().compute_dtype)
#     outputs = {}
#     if self.cnn_keys:
#       outputs.update(self._cnn(features))
#     if self.mlp_keys:
#       outputs.update(self._mlp(features))
#     return outputs

#   def _cnn(self, features):
#     """
#       (16, 10, deter + num_tokens * stoch_size)
#       (16, 10, hiddim) --> the discrete latents select codebook vectors and sum them
#       (160, 1, 1, hiddim)  
#       --> start with a 1x1, and then you end up distributing that across space.

#       0 (B, 5, 5, 16)
#       1 (B, 13, 13, 8)
#       2 (B, 30, 30, 4)
#       3 (B, 64, 64, 3)

#       actual:

#       features: (B, T, deter + num_tokens * stoch_size)
#       x: (B, T, 1536)
#       x: (B*T, 1, 1, 1536)
#       x: (B*T, 5, 5, 192)
#       x: (B*T, 13, 13, 96)
#       x: (B*T, 30, 30, 48)
#       x: (B*T, 64, 64, 3)
#       x: (B, T, 64, 64, 3)
#       means: [(B, T, 64, 64, 3)]
#     """
#     channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
#     ConvT = tfkl.Conv2DTranspose
#     x = self.get('convin', tfkl.Dense, 32 * self._cnn_depth)(features)
#     x = eo.rearrange(x, '... d -> (...) 1 1 d')
#     for i, kernel in enumerate(self._cnn_kernels):
#       depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
#       act, norm = self._act, self._norm
#       if i == len(self._cnn_kernels) - 1:
#         depth, act, norm = sum(channels.values()), tf.identity, 'none'
#       x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
#       x = self.get(f'convnorm{i}', NormLayer, norm)(x)
#       x = act(x)
#     x = x.reshape(features.shape[:-1] + x.shape[1:])  # (B, T, H, W, C)
#     means = tf.split(x, list(channels.values()), -1)  # [(B, T, H, W, C)]
#     dists = {
#         key: tfd.Independent(tfd.Normal(mean, 1), 3)
#         for (key, shape), mean in zip(channels.items(), means)}
#     return dists

#   def _mlp(self, features):
#     shapes = {k: self._shapes[k] for k in self.mlp_keys}
#     x = features
#     for i, width in enumerate(self._mlp_layers):
#       x = self.get(f'dense{i}', tfkl.Dense, width)(x)
#       x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
#       x = self._act(x)
#     dists = {}
#     for key, shape in shapes.items():
#       dists[key] = self.get(f'dense_{key}', DistLayer, shape)(x)
#     return dists


class GridDecoder(Decoder):
  def __init__(self, shapes, decoder_type, pos_encode_type, token_dim, resolution, **kwargs):
    super().__init__(shapes, **kwargs)

    self.resolution = resolution
    self.token_dim = token_dim

    decoder_type, transformer_type = split_at_n(decoder_type, '_', 2)

    if decoder_type == 'grid_g':
      self.decoder = dvae.GenericDecoder(in_channels=self.token_dim, out_channels=3)
    elif decoder_type == 'grid_dvweak':
      self.decoder = dvae.dVAEShallowWeakDecoder(in_channels=self.token_dim, out_channels=3)
      # TODO: you need to add position embedding to this! 
    elif decoder_type == 'grid_dvstrong':
      self.decoder = dvae.dVAEStrongDecoder(in_channels=self.token_dim, out_channels=3)
      # TODO: you need to add position embedding to this! 
    elif decoder_type == 'grid_sa':
      pass
    elif decoder_type == 'grid_saslim':
      pass
    elif decoder_type == 'grid_sadebug':
      pass
    else:
      raise NotImplementedError

    if pos_encode_type == 'slate':
      self.position_encoding = transformer.GridPositionalEncoding(
        resolution=self.resolution, dim=self.token_dim)
    elif pos_encode_type == 'coordconv':
      self.position_encoding = transformer.CoordConvPositionalEncoding(
        resolution=self.resolution, dim=self.token_dim)
    elif pos_encode_type == 'sinusoid':
      pass
    elif pos_encode_type == 'none':
      self.position_encoding = lambda x: x
    else:
      raise NotImplementedError

    # self.tf_dec = transformer.TransformerDecoder(self.token_dim, transformer.TransformerDecoder.obs_cross_defaults())
    # self.tf_dec = transformer.TransformerDecoder(self.token_dim, transformer.TransformerDecoder.two_blocks_eight_heads_defaults())

    if transformer_type == 'dec':
      self.tf_dec = transformer.TransformerDecoder(self.token_dim, transformer.TransformerDecoder.two_blocks_four_heads_defaults())
    elif transformer_type == 'ca':
      self.tf_dec = transformer.CrossAttentionStack(self.token_dim, transformer.TransformerDecoder.two_blocks_four_heads_defaults())
    else:
      raise NotImplementedError
    # self.tf_dec = transformer.TransformerDecoder(self.token_dim, transformer.TransformerDecoder.two_blocks_four_heads_defaults())
    # self.tf_dec = transformer.CrossAttentionStack(self.token_dim, transformer.TransformerDecoder.two_blocks_four_heads_defaults())

    self.token_mlp = tf.keras.Sequential([
        tfkl.Dense(self.token_dim, kernel_initializer='he_uniform'),
        tfkl.ReLU(),
        tfkl.Dense(self.token_dim, kernel_initializer='he_uniform')])

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

      actual:

      features: (B, T, deter + num_tokens * stoch_size)
      x: (B, T, 1536)
      x: (B*T, 1, 1, 1536)
      x: (B*T, 5, 5, 192)
      x: (B*T, 13, 13, 96)
      x: (B*T, 30, 30, 48)
      x: (B*T, 64, 64, 3)
      x: (B, T, 64, 64, 3)
      means: [(B, T, 64, 64, 3)]

      when slot-based:
      
    """
    # import ipdb; ipdb.set_trace(context=20)
    channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
    #############################################################
    batch_dims = features.shape[:2]  # (B, T) or (H, B*T)
    x = self.get('convin', tfkl.Dense, self.token_dim)(features)

    # 1. reshape features into slots
    x = eo.rearrange(x, '... k d -> (...) k d')
    # 2. create queries by applying position encodings to zeros, then token_mlp
    bsize = x.shape[0]
    queries = tf.zeros([bsize] + list(self.resolution) + [self.token_dim], dtype=x.dtype)
    queries = self.token_mlp(self.position_encoding(queries))
    queries = eo.rearrange(queries, '... h w d -> ... (h w) d')
    # 3. tf_dec --> (16x16)
    grid = self.tf_dec(queries, x)
    grid = eo.rearrange(grid, '... (h w) d -> ... h w d', h=self.resolution[0], w=self.resolution[1])
    # 4. apply cnn decoder
    x = self.decoder(grid)
    x = x.reshape(batch_dims + x.shape[1:])  # (B, T, H, W, C)

    # ***********************************************************
    # ConvT = tfkl.Conv2DTranspose
    # x = self.get('convin', tfkl.Dense, 32 * self._cnn_depth)(features)
    # x = eo.rearrange(x, '... d -> (...) 1 1 d')
    # for i, kernel in enumerate(self._cnn_kernels):
    #   depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
    #   act, norm = self._act, self._norm
    #   if i == len(self._cnn_kernels) - 1:
    #     depth, act, norm = sum(channels.values()), tf.identity, 'none'
    #   x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
    #   x = self.get(f'convnorm{i}', NormLayer, norm)(x)
    #   x = act(x)
    # x = x.reshape(features.shape[:-1] + x.shape[1:])  # (B, T, H, W, C)
    #############################################################
    means = tf.split(x, list(channels.values()), -1)  # [(B, T, H, W, C)]
    dists = {
        key: tfd.Independent(tfd.Normal(mean, 1), 3)
        for (key, shape), mean in zip(channels.items(), means)}
    return dists


    # if return_components:
    #   masked_components = eo.rearrange(masked_components, '(b t) ... -> b t ...', b=batch_size)
    #   masked_components = nmlz.center(masked_components)
    #   dists['components'] = tfd.Independent(tfd.Normal(masked_components, 1), 3)  # (b t k h w c)
    #   dists['masks'] = masks



"""
TODO:
  - make the output shape of the decoder configurable
  - take out reshaping for slots
"""
class PreviousSlotDecoder(Decoder):
  def __init__(self, shapes, indim, decoder_type, **kwargs):
    super().__init__(shapes, **kwargs)
    self.indim = indim
    if decoder_type == 'slot':
      self.decoder = slot_attention.SlotAttentionDecoder(indim, (64, 64))  # hardcoded to (64, 64) with indim 112
    elif decoder_type == 'slimmerslot':
      self.decoder = slot_attention.DebugSlotAttentionDecoder6464(indim)  # hardcoded to (64, 64) with indim 112
    else:
      raise NotImplementedError

  def __call__(self, features, return_components=False):
    features = tf.cast(features, prec.global_policy().compute_dtype)
    outputs = {}
    if self.cnn_keys:
      outputs.update(self._cnn(features, return_components))
    if self.mlp_keys:
      outputs.update(self._mlp(features))
    return outputs

  def _cnn(self, features, return_components=False):
    """
      features: (B, T, D)
    """
    channels = {k: self._shapes[k][-1] for k in self.cnn_keys}

    # hacky reshape for now
    batch_size, seq_length = features.shape[:2]
    slots = eo.rearrange(features, 'b t (k d) -> (b t) k d', d=self.indim)
    x = self.decoder(slots)

    # Undo combination of slot and batch dimension; split alpha masks.
    recons, masks = slot_attention.unstack_and_split(x, batch_size=batch_size*seq_length)
    # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    masks = tf.nn.softmax(masks, axis=1)

    # uncenter the components
    recons = nmlz.uncenter(recons)
    masked_components = recons * masks

    # combine
    recon_combined = eo.reduce(masked_components, 'bt k h w c -> bt h w c', 'sum')  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].

    # center the combination
    recon_combined = nmlz.center(recon_combined)

    x = eo.rearrange(recon_combined, '(b t) ... -> b t ...', b=batch_size)
    #############################################################
    means = tf.split(x, list(channels.values()), -1)  # [(B, T, H, W, C)]
    dists = {
        key: tfd.Independent(tfd.Normal(mean, 1), 3)
        for (key, shape), mean in zip(channels.items(), means)}

    if return_components:
      masked_components = eo.rearrange(masked_components, '(b t) ... -> b t ...', b=batch_size)
      masked_components = nmlz.center(masked_components)
      dists['components'] = tfd.Independent(tfd.Normal(masked_components, 1), 3)  # (b t k h w c)
      dists['masks'] = masks

    return dists


#############################################################
# Layers
#############################################################

# class MLP(common.Module):

#   def __init__(self, shape, layers, units, act='elu', norm='none', **out):
#     self._shape = (shape,) if isinstance(shape, int) else shape
#     self._layers = layers
#     self._units = units
#     self._norm = norm
#     self._act = get_act(act)
#     self._out = out

#   def __call__(self, features):
#     x = tf.cast(features, prec.global_policy().compute_dtype)
#     x = x.reshape([-1, x.shape[-1]])
#     for index in range(self._layers):
#       x = self.get(f'dense{index}', tfkl.Dense, self._units)(x)
#       x = self.get(f'norm{index}', NormLayer, self._norm)(x)
#       x = self._act(x)
#     x = x.reshape(features.shape[:-1] + [x.shape[-1]])
#     return self.get('out', DistLayer, self._shape, **self._out)(x)


# class GRUCell(tf.keras.layers.AbstractRNNCell):

#   def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
#     super().__init__()
#     self._size = size
#     self._act = get_act(act)
#     self._norm = norm
#     self._update_bias = update_bias
#     self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
#     if norm:
#       self._norm = tfkl.LayerNormalization(dtype=tf.float32)

#   @property
#   def state_size(self):
#     return self._size

#   @tf.function
#   def call(self, inputs, state):
#     state = state[0]  # Keras wraps the state in a list.
#     parts = self._layer(tf.concat([inputs, state], -1))
#     if self._norm:
#       dtype = parts.dtype
#       parts = tf.cast(parts, tf.float32)
#       parts = self._norm(parts)
#       parts = tf.cast(parts, dtype)
#     reset, cand, update = tf.split(parts, 3, -1)
#     reset = tf.nn.sigmoid(reset)
#     cand = self._act(reset * cand)
#     update = tf.nn.sigmoid(update + self._update_bias)
#     output = update * cand + (1 - update) * state
#     return output, [output]


# class DistLayer(common.Module):

#   def __init__(
#       self, shape, dist='mse', min_std=0.1, init_std=0.0):
#     self._shape = shape
#     self._dist = dist
#     self._min_std = min_std
#     self._init_std = init_std

#   def __call__(self, inputs):
#     out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
#     out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
#     out = tf.cast(out, tf.float32)
#     if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
#       std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
#       std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
#       std = tf.cast(std, tf.float32)
#     if self._dist == 'mse':
#       dist = tfd.Normal(out, 1.0)
#       return tfd.Independent(dist, len(self._shape))
#     if self._dist == 'normal':
#       dist = tfd.Normal(out, std)
#       return tfd.Independent(dist, len(self._shape))
#     if self._dist == 'binary':
#       dist = tfd.Bernoulli(out)
#       return tfd.Independent(dist, len(self._shape))
#     if self._dist == 'tanh_normal':
#       mean = 5 * tf.tanh(out / 5)
#       std = tf.nn.softplus(std + self._init_std) + self._min_std
#       dist = tfd.Normal(mean, std)
#       dist = tfd.TransformedDistribution(dist, common.TanhBijector())
#       dist = tfd.Independent(dist, len(self._shape))
#       return common.SampleDist(dist)
#     if self._dist == 'trunc_normal':
#       std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
#       dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
#       return tfd.Independent(dist, 1)
#     if self._dist == 'onehot':
#       return common.OneHotDist(out)
#     raise NotImplementedError(self._dist)


# class NormLayer(common.Module):

#   def __init__(self, name):
#     if name == 'none':
#       self._layer = None
#     elif name == 'layer':
#       self._layer = tfkl.LayerNormalization()
#     else:
#       raise NotImplementedError(name)

#   def __call__(self, features):
#     if not self._layer:
#       return features
#     return self._layer(features)


# def get_act(name):
#   if name == 'none':
#     return tf.identity
#   if name == 'mish':
#     return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
#   elif hasattr(tf.nn, name):
#     return getattr(tf.nn, name)
#   elif hasattr(tf, name):
#     return getattr(tf, name)
#   else:
#     raise NotImplementedError(name)


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
    """
      prev_deter: (B, deter_dim)
      prev_stoch: (B, stoch, discrete)
      prev_action: (B, action_dim)
    """
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', NormLayer, self._norm)(x)  # why do they normalize after the linear?
    x = self._act(x)
    x, deter = self._cell(x, [prev_deter])
    deter = deter[0]  # Keras wraps the state in a list.
    return deter, deter


class CrossDynamics(common.Module):
  def __init__(self, deter, hidden, act, norm):
    self._deter = deter
    self._hidden = hidden
    self._act = act
    self._norm = norm

    # just to get the initial state for now
    self._cell = GRUCell(self._deter, norm=True)

    self.net = transformer.TransformerDecoder(
      self._hidden, 
      transformer.TransformerDecoder.one_block_one_head_defaults()
      )

  def __call__(self, prev_deter, prev_stoch, prev_action):
    """
      prev_deter: (B, K, deter_dim)
      prev_stoch: (B, K, num_stoch, stoch_dim)
      prev_action: (B, A)
    """
    stoch_embed = self.get('stoch_embed', tfkl.Dense, self._hidden)(prev_stoch)
    act_embed =  self.get('act_embed', tfkl.Dense, self._hidden, use_bias=False)(unflatten(prev_action, 1))
    context = tf.concat([stoch_embed, act_embed], 1)  # (B, K+1, H)
    deter = self.net(prev_deter, context)
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

    self._cell = GRUCell(self._deter//self.num_slots, norm=True)
    self.context_attention = attention.ContextAttention(self._deter//self.num_slots, num_heads=1)

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots

  def __call__(self, prev_deter, prev_stoch, prev_action):
    stoch_embed = self.get('stoch_embed', tfkl.Dense, self._hidden)(prev_stoch)  # (B, K, S*V) --> (B, K, H)
    act_embed =  self.get('act_embed', tfkl.Dense, self._hidden)(prev_action)  # (B, A) --> (B, H)
    context = tf.concat([stoch_embed, eo.rearrange(act_embed, 'b a -> b 1 a')], 1)  # (B, K+1, H)

    x = self.context_attention(prev_deter, context, mask=None)  # (B, K, D)
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
    """
      deter: (B, deter_dim)
      embed: (B, embed_dim)
    """
    x = tf.concat([deter, embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
    x = self.get('obs_out_norm', NormLayer, self._norm)(x)  # why do they normalize after the linear?
    x = self._act(x)
    return x


class CrossUpdate(common.Module):
  def __init__(self, hidden, act, norm):
    self._hidden = hidden
    self._act = act
    self._norm = norm

    self.net = transformer.TransformerDecoder(
      self._hidden, 
      transformer.TransformerDecoder.one_block_one_head_defaults()
      )

  def __call__(self, deter, embed):
    """
      deter: (B, deter_dim)
      embed: (B, S, embed_dim)
    """
    num_slots = 1
    deter = unflatten(deter, num_slots)
    x = self.net(deter, embed)
    x = self._act(x)
    x = flatten(x)  # take out later
    return x


# class SlotUpdate(common.Module):
#   @staticmethod
#   def defaults():
#       default_args = slot_attn.SlotAttention.savi_defaults()
#       return default_args

#   def __init__(self, hidden, act, norm, slot_config):
#     self._hidden = hidden
#     self._act = act
#     self._norm = norm

#     # self.slot_attn = slot_attn.SlotAttention(
#     #   self._hidden, 
#     #   slot_attn.SlotAttention.savi_defaults())

#     self.slot_attn = slot_attn.SlotAttention(
#       self._hidden, slot_config)
#       # slot_attn.SlotAttention.savi_defaults())

#   def __call__(self, deter, embed, return_attns=False):
#     """
#       deter: (B, K, deter_dim)
#       embed: (B, K, S, embed_dim)
#     """
#     x, attns = self.slot_attn(embed, deter)
#     if return_attns:
#       return x, attns
#     else:
#       return x

class SlotUpdate(common.Module):
  @staticmethod
  def defaults():
      default_args = slot_attn.SlotAttention.savi_defaults()
      return default_args

  def __init__(self, hidden, act, norm):
    self._hidden = hidden
    self._act = act
    self._norm = norm

    self.slot_attn = slot_attn.SlotAttention(
      self._hidden, 
      slot_attn.SlotAttention.savi_defaults())

    # self.slot_attn = slot_attn.SlotAttention(
    #   self._hidden, slot_config)
    #   # slot_attn.SlotAttention.savi_defaults())

  def __call__(self, deter, embed, return_attns=False):
    """
      deter: (B, K, deter_dim)
      embed: (B, K, S, embed_dim)
    """
    x, attns = self.slot_attn(embed, deter)
    if return_attns:
      return x, attns
    else:
      return x


class SlimAttentionUpdate(common.Module):
  def __init__(self, deter, act, norm, embed_dim, num_slots):
    self._deter = deter
    self._act = act
    self._norm = norm
    self._embed_dim = embed_dim

    self.num_slots = num_slots

    self._cell = GRUCell(self._deter//self.num_slots, norm=True)
    self.context_attention = attention.ContextAttention(self._deter//self.num_slots, num_heads=1)

  # def register_num_slots(self, num_slots):
  #   self.num_slots = num_slots

  def __call__(self, deter, embed):
    context = eo.rearrange(embed, 'b (gridsize embed_dim) -> b gridsize embed_dim', embed_dim=self._embed_dim)  
    x = self.context_attention(deter, context, mask=None)
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
      # num_iterations=3, 
      slot_size=self._deter//self.num_slots, 
      # mlp_hidden_size=self._deter//self.num_slots,
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
    context = eo.rearrange(embed, 'b (gridsize embed_dim) -> b gridsize embed_dim', embed_dim=self._embed_dim)  
    updated_slots = self.slot_attention(deter, context)
    return updated_slots

