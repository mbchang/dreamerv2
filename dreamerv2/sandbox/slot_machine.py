from loguru import logger as lgr
import re
import einops as eo
import ml_collections
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

  # TODO: create your default config here!

  @staticmethod
  def defaults():
      default_args = ml_collections.ConfigDict(dict(
        cross_dynamics=CrossDynamics.defaults(),
        slot_update=SlotUpdate.defaults(),
        cross_update=CrossUpdate.defaults(),
          ))
      return default_args

  def __init__(self, config, slot_config, resolution):
    common.Module.__init__(self)
    self._ensemble = config.ensemble
    self._stoch = config.stoch
    self._deter = config.deter
    self._hidden = config.hidden
    self._discrete = config.discrete
    self._act = get_act(config.act)
    self._norm = config.norm
    self._std_act = config.std_act
    self._min_std = config.min_std
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    self._dynamics_type = config.dynamics_type
    self._update_type = config.update_type
    # self._embed_dim = config.embed_dim
    self._initial_type = config.initial_type
    self.num_slots = config.num_slots
    self._resolution = resolution

    if self._dynamics_type == 'default':
      self.dynamics = DefaultDynamics(self._deter, self._hidden, self._act, self._norm)
    elif self._dynamics_type == 'cross':
      self.dynamics = CrossDynamics(self._deter, self._hidden, self._act, self._norm, slot_config.cross_dynamics)
    else:
      raise NotImplementedError

    if self._update_type == 'default':
      self.update = DefaultUpdate(self._hidden, self._act, self._norm)
    elif self._update_type == 'cross':
      self.update = CrossUpdate(self._hidden, self._act, self._norm, slot_config.cross_update)
    elif self._update_type == 'slot':
      self.update = SlotUpdate(self._hidden, self._act, self._norm, slot_config.slot_update)
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

class GridEncoder(Encoder):
  @staticmethod
  def defaults():
      default_args = ml_collections.ConfigDict(dict(
        encoder_type='grid_g',
          ))
      return default_args

  # def __init__(self, shapes, pos_encode_type, outdim, resolution, slot_config,**kwargs):
  def __init__(self, shapes, obs_itf, slot_config,**kwargs):
    super().__init__(shapes, **kwargs)

    self._token_dim = obs_itf.token_dim
    self._pos_encode_type = obs_itf.pos_encode_type
    self._resolution = obs_itf.resolution
    self._encoder_type = slot_config.encoder_type

    if self._encoder_type == 'grid_g':
      self.encoder = dvae.GenericEncoder(in_channels=3, out_channels=self._token_dim)
    elif self._encoder_type == 'grid_dvweak':
      self.encoder = dvae.dVAEShallowWeakEncoder(in_channels=3, out_channels=self._token_dim)
      # TODO: you need to add position embedding to this! 
    elif self._encoder_type == 'grid_dvstrong':
      self.encoder = dvae.dVAEStrongEncoder(in_channels=3, out_channels=self._token_dim)
      # TODO: you need to add position embedding to this! 
    elif self._encoder_type == 'grid_sa':
      pass
    elif self._encoder_type == 'grid_saslim':
      pass
    elif self._encoder_type == 'grid_sadebug':
      pass
    else:
      raise NotImplementedError

    if self._pos_encode_type == 'slate':
      self.position_encoding = transformer.GridPositionalEncoding(
        resolution=self._resolution, dim=self._token_dim)
    elif self._pos_encode_type == 'coordconv':
      self.position_encoding = transformer.CoordConvPositionalEncoding(
        resolution=self._resolution, dim=self._token_dim)
    elif self._pos_encode_type == 'sinusoid':
      pass
    elif self._pos_encode_type == 'none':
      self.position_encoding = lambda x: x
    else:
      raise NotImplementedError

    self.token_mlp = tf.keras.Sequential([
        tfkl.Dense(self._token_dim, kernel_initializer='he_uniform'),
        tfkl.ReLU(),
        tfkl.Dense(self._token_dim, kernel_initializer='he_uniform')])


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


class GridDecoder(Decoder):
  @staticmethod
  def defaults():
      default_args = ml_collections.ConfigDict(dict(
          decoder_type='grid_g',
          transformer_type='ca',

          dec_config=transformer.TransformerDecoder.two_blocks_four_heads_defaults(),
          ca_config=transformer.TransformerDecoder.two_blocks_four_heads_defaults(),
          ))
      return default_args

  # def __init__(self, shapes, pos_encode_type, token_dim, resolution, slot_config, **kwargs):
  def __init__(self, shapes, obs_itf, slot_config, **kwargs):
    super().__init__(shapes, **kwargs)

    self._resolution = obs_itf.resolution
    self._token_dim = obs_itf.token_dim
    self._pos_encode_type = obs_itf.pos_encode_type
    self._decoder_type = slot_config.decoder_type
    self._transformer_type = slot_config.transformer_type

    if self._decoder_type == 'grid_g':
      self.decoder = dvae.GenericDecoder(in_channels=self._token_dim, out_channels=3)
    elif self._decoder_type == 'grid_dvweak':
      self.decoder = dvae.dVAEShallowWeakDecoder(in_channels=self._token_dim, out_channels=3)
      # TODO: you need to add position embedding to this! 
    elif self._decoder_type == 'grid_dvstrong':
      self.decoder = dvae.dVAEStrongDecoder(in_channels=self._token_dim, out_channels=3)
      # TODO: you need to add position embedding to this! 
    elif self._decoder_type == 'grid_sa':
      pass
    elif self._decoder_type == 'grid_saslim':
      pass
    elif self._decoder_type == 'grid_sadebug':
      pass
    else:
      raise NotImplementedError

    if self._pos_encode_type == 'slate':
      self.position_encoding = transformer.GridPositionalEncoding(
        resolution=self._resolution, dim=self._token_dim)
    elif self._pos_encode_type == 'coordconv':
      self.position_encoding = transformer.CoordConvPositionalEncoding(
        resolution=self._resolution, dim=self._token_dim)
    elif self._pos_encode_type == 'sinusoid':
      pass
    elif self._pos_encode_type == 'none':
      self.position_encoding = lambda x: x
    else:
      raise NotImplementedError

    # self.tf_dec = transformer.TransformerDecoder(self.token_dim, transformer.TransformerDecoder.obs_cross_defaults())
    # self.tf_dec = transformer.TransformerDecoder(self.token_dim, transformer.TransformerDecoder.two_blocks_eight_heads_defaults())

    if self._transformer_type == 'dec':
      self.tf_dec = transformer.TransformerDecoder(self._token_dim, slot_config.dec_config)
    elif self._transformer_type == 'ca':
      self.tf_dec = transformer.CrossAttentionStack(self._token_dim, slot_config.ca_config)
    else:
      raise NotImplementedError
    # self.tf_dec = transformer.TransformerDecoder(self.token_dim, transformer.TransformerDecoder.two_blocks_four_heads_defaults())
    # self.tf_dec = transformer.CrossAttentionStack(self.token_dim, transformer.TransformerDecoder.two_blocks_four_heads_defaults())

    self.token_mlp = tf.keras.Sequential([
        tfkl.Dense(self._token_dim, kernel_initializer='he_uniform'),
        tfkl.ReLU(),
        tfkl.Dense(self._token_dim, kernel_initializer='he_uniform')])

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
    x = self.get('convin', tfkl.Dense, self._token_dim)(features)

    # 1. reshape features into slots
    x = eo.rearrange(x, '... k d -> (...) k d')
    # 2. create queries by applying position encodings to zeros, then token_mlp
    bsize = x.shape[0]
    queries = tf.zeros([bsize] + list(self._resolution) + [self._token_dim], dtype=x.dtype)
    queries = self.token_mlp(self.position_encoding(queries))
    queries = eo.rearrange(queries, '... h w d -> ... (h w) d')
    # 3. tf_dec --> (16x16)
    grid = self.tf_dec(queries, x)
    grid = eo.rearrange(grid, '... (h w) d -> ... h w d', h=self._resolution[0], w=self._resolution[1])
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


#############################################################
# Dynamics
#############################################################
class CrossDynamics(common.Module):
  @staticmethod
  def defaults():
      default_args = transformer.TransformerDecoder.one_block_one_head_defaults()
      return default_args

  def __init__(self, deter, hidden, act, norm, slot_config):
    self._deter = deter
    self._hidden = hidden
    self._act = act
    self._norm = norm

    # just to get the initial state for now
    self._cell = GRUCell(self._deter, norm=True)

    self.net = transformer.TransformerDecoder(self._hidden, slot_config)
      # transformer.TransformerDecoder.one_block_one_head_defaults()
      # )

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


#############################################################
# Update
#############################################################

class CrossUpdate(common.Module):
  @staticmethod
  def defaults():
      default_args = transformer.TransformerDecoder.one_block_one_head_defaults()
      return default_args

  def __init__(self, hidden, act, norm, slot_config):
    self._hidden = hidden
    self._act = act
    self._norm = norm

    self.net = transformer.TransformerDecoder(self._hidden, slot_config)

  def __call__(self, deter, embed):
    """
      deter: (B, deter_dim)
      embed: (B, S, embed_dim)
    """
    x = self.net(deter, embed)
    x = self._act(x)
    return x


class SlotUpdate(common.Module):
  @staticmethod
  def defaults():
      default_args = slot_attn.SlotAttention.savi_defaults()
      return default_args

  def __init__(self, hidden, act, norm, slot_config):
    self._hidden = hidden
    self._act = act
    self._norm = norm
    self.slot_attn = slot_attn.SlotAttention(self._hidden, slot_config)

  def __call__(self, deter, embed, return_attns=False):
    """
      deter: (B, K, deter_dim)
      embed: (B, K, S, embed_dim)
    """
    x, attns = self.slot_attn(embed, deter)
    # TODO: should you apply an activation here?
    if return_attns:
      return x, attns
    else:
      return x
