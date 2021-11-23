import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import hashlib
from loguru import logger as lgr
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow.keras.activations as tka
import tensorflow_addons as tfa


########################################################################
## Training utils
########################################################################
def linear_warmup(step, start_value, final_value, start_step, final_step):
    
    assert start_value <= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b
    
    return value


def cosine_anneal(step, start_value, final_value, start_step, final_step):
    
    assert start_value >= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b
    
    return value


def f32(x):
    return tf.cast(x, tf.float32)


########################################################################
## Visualization utils
########################################################################


def overlay_attention(attns, image):
    *_, H, W = image.shape
    attns = rearrange(attns, 'b hw k -> b k hw')
    size = int(np.sqrt(attns.shape[-1]))
    attns = tf.repeat(tf.repeat(rearrange(attns, 'b k (h w) -> b k 1 h w', h = size, w=size), H // size, axis=-2), W // size, axis=-1)
    attns = image * attns + 1. - attns
    return attns


def report(image, attns, recon, model, preproc):
    """
    Ideally this should just take the data as input only
    """
    gen_img = model.reconstruct_autoregressive(image)
    unsqueeze = lambda x: rearrange(preproc(x), 'b c h w -> b 1 c h w')
    vis_recon = tf.concat((
        unsqueeze(image), 
        unsqueeze(recon), 
        unsqueeze(gen_img), 
        overlay_attention(attns, unsqueeze(image))), axis=1)
    return vis_recon

def hash_10(obj):
    return hashlib.shake_128(str(obj).encode('utf-8')).hexdigest(10)


########################################################################
## Algorithm utils
########################################################################

def bottle(fn):
  """
    1. x: b t ...
    2. x: (b t) ...
    3. fn(x): (b t) ...
    4. fn(x): b t ...
  """
  def bottled_fn(*x):
    x = tuple(x)
    n_args = len(x)
    bsize = len(x[0])

    flatten = lambda z: rearrange(z, 'b t ... -> (b t) ...', b=bsize)
    unflatten = lambda z: rearrange(z, '(b t) ... -> b t ...', b=bsize)

    if n_args > 1:
        y = fn(*tuple(flatten(xx) for xx in x))
    else:
        y = fn(flatten(x[0]))

    if isinstance(y, tuple):
      return tuple(unflatten(yy) for yy in y)
    else:
      return unflatten(y)
  return bottled_fn

########################################################################
## Network utils
########################################################################


def gumbel_max(logits, dim=-1):
    
    eps = torch.finfo(logits.dtype).tiny
    
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels
    
    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    
    eps = tf.experimental.numpy.finfo(logits.dtype).tiny
    
    gumbels = -tf.math.log(tf.random.gamma(logits.shape, alpha=1, beta=1) + eps)
    gumbels = (logits + gumbels) / tau
    
    y_soft = tf.nn.softmax(gumbels, axis=dim)
    
    if hard:
        index = tf.math.argmax(y_soft, axis=dim)
        y_hard = tf.one_hot(index, depth=logits.shape[dim], axis=dim)
        return y_hard - tf.stop_gradient(y_soft) + y_soft
    else:
        return y_soft


def log_prob_gaussian(value, mean, std):
    
    var = std ** 2
    if isinstance(var, float):
        return -0.5 * (((value - mean) ** 2) / var + math.log(var) + math.log(2 * math.pi))
    else:
        return -0.5 * (((value - mean) ** 2) / var + var.log() + math.log(2 * math.pi))


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    
    m = tf.keras.Sequential([
        tkl.ZeroPadding2D(padding=padding),
        tkl.Conv2D(
            filters=out_channels, 
            kernel_size=kernel_size, 
            strides=(stride, stride), 
            dilation_rate=(dilation, dilation),
            groups=groups,
            use_bias=bias,
            kernel_initializer='he_uniform' if weight_init == 'kaiming' else 'glorot_uniform')
        ])

    return m

class Conv2dBlock(tkl.Layer):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=False, weight_init='kaiming')
        self.group_norm = tfa.layers.GroupNormalization(groups=1, axis=-1)
    
    def call(self, x):
        x = self.m(x)
        x = tka.relu(self.group_norm(x))
        return x

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    if weight_init == 'kaiming':
        kernel_initializer = 'he_uniform'
    else:
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=gain**2, 
            mode='fan_avg', distribution='uniform')

    m = tkl.Dense(
        units=out_features, 
        use_bias=bias,
        kernel_initializer=kernel_initializer)
        
    return m


def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m

if __name__ == '__main__':
    import numpy as np
    x = np.random.uniform(size=(5, 3, 64, 64))
    m = Conv2dBlock(3, 64, 4, 4)
    y = m(x)
    print('y', y.shape)