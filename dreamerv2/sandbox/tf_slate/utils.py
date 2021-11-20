import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow.keras.activations as tka
import tensorflow_addons as tfa


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
        y = fn(flatten(x))

    if isinstance(y, tuple):
      return tuple(unflatten(yy) for yy in y)
    else:
      return unflatten(y)
  return bottled_fn


def gumbel_max(logits, dim=-1):
    
    eps = torch.finfo(logits.dtype).tiny
    
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels
    
    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    
    # eps = torch.finfo(logits.dtype).tiny
    eps = tf.experimental.numpy.finfo(logits.dtype).tiny
    
    # gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = -tf.math.log(tf.random.gamma(logits.shape, alpha=1, beta=1) + eps)
    gumbels = (logits + gumbels) / tau
    
    # y_soft = F.softmax(gumbels, dim)
    y_soft = tf.nn.softmax(gumbels, axis=dim)
    
    if hard:
        # index = y_soft.argmax(dim, keepdim=True)
        index = tf.math.argmax(y_soft, axis=dim)
        # y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        y_hard = tf.one_hot(index, depth=logits.shape[dim], axis=dim)
        # return y_hard - y_soft.detach() + y_soft
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
    
    # m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
    #               dilation, groups, bias, padding_mode)
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

    
    # if weight_init == 'kaiming':
    #     nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    # else:
    #     nn.init.xavier_uniform_(m.weight)
    
    # if bias:
    #     nn.init.zeros_(m.bias)

    return m

class Conv2dBlock(tkl.Layer):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=False, weight_init='kaiming')
        # self.weight = nn.Parameter(torch.ones(out_channels))
        # self.bias = nn.Parameter(torch.zeros(out_channels))
        self.group_norm = tfa.layers.GroupNormalization(groups=1, axis=-1)
    
    
    def call(self, x):
        # TODO: later we will just reshape the input once
        x = self.m(x)
        x = tka.relu(self.group_norm(x))
        return x
        # return F.relu(F.group_norm(x, 1, self.weight, self.bias))

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    # m = nn.Linear(in_features, out_features, bias)
    if weight_init == 'kaiming':
        # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        kernel_initializer = 'he_uniform'
    else:
        # nn.init.xavier_uniform_(m.weight, gain)
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=gain**2, 
            mode='fan_avg', distribution='uniform')

    m = tkl.Dense(
        units=out_features, 
        use_bias=bias,
        kernel_initializer=kernel_initializer)
    
    # if bias:
    #     nn.init.zeros_(m.bias)
    
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