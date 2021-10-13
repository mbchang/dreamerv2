import numpy as np
import tensorflow as tf

import common


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  # dk = tf.cast(tf.shape(k)[-1], tf.float32)
  dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output

class MultiHeadAttention(common.Module):
  def __init__(self, d_model, num_heads):
    # super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def __call__(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output

class ContextAttention(MultiHeadAttention):
    def __call__(self, x, c, mask):
      return super().__call__(c, c, x, mask)


def point_wise_feed_forward_network(dim):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dim*4, activation='gelu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(dim)  # (batch_size, seq_len, d_model)
  ])

class CrossAttentionBlock(common.Module):
  def __init__(self, dim, num_heads, rate=0.1):
    super(CrossAttentionBlock, self).__init__()

    self.mha = ContextAttention(dim, num_heads)
    self.ffn = point_wise_feed_forward_network(dim)

    self.lnq1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.lnc = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.lnq2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

    self.train = True

  def train(self):
    self.train = True

  def eval(self):
    self.train = False
    
  def __call__(self, x, c, mask=None):
    assert len(x.shape) == 3 and len(c.shape) == 3

    attn_output = self.mha(self.lnq1(x), self.lnc(c), mask)
    x = x + self.dropout1(attn_output, training=self.train)
    
    ffn_output = self.ffn(self.lnq2(x))
    x = x + self.dropout2(ffn_output, training=self.train)
    
    return x


if __name__ == '__main__':
    tf.random.set_seed(0)
    np.random.seed(0)

    # embed_dim = 8
    # num_heads = 4
    # batch_size = 2
    # K = 5

    # cab = CrossAttentionBlock(embed_dim, num_heads)

    # x = tf.random.uniform((batch_size, K, embed_dim))
    # y = cab(x, x)

    # print(y.shape)  # (batch_size, K, embed_dim)
    # print(y)


    embed_dim = 48
    num_heads = 4
    batch_size = 10
    K = 1

    cabs = [CrossAttentionBlock(embed_dim, num_heads, rate=0) for i in range(100)]
    # cab = CrossAttentionBlock(embed_dim, num_heads)

    x = tf.zeros((batch_size, K, embed_dim))
    c = tf.random.uniform((batch_size, K, embed_dim))

    print(tf.norm(c, axis=-1))
    # for i in range(10):
    for cab in cabs:
      x = cab(x, c)
      print(x.shape)  # (batch_size, K, embed_dim)
      print(tf.norm(x, axis=-1))


