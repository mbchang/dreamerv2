# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slot Attention model for object discovery and set prediction."""
from einops import rearrange, repeat
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

class Factorized(tf.Module):
  def register_num_slots(self, num_slots):
    self.num_slots = num_slots
    for sm in self.submodules:
      if isinstance(sm, Factorized):
        sm.register_num_slots(num_slots)

class SlotAttention(layers.Layer, Factorized):
  """Slot Attention module."""
  epsilon = 1e-8
  num_iterations = 3

  @staticmethod
  def get_default_args():
    default_args = ml_collections.ConfigDict(
      dict(

        ))
    return default_args

  def __init__(self, slot_size, temp=1, learn_initial_dist=True):
    """Builds the Slot Attention module.

    Args:
      num_iterations: Number of iterations.
      num_slots: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      epsilon: Offset for attention coefficients before normalization.
    """
    super().__init__()
    self.slot_size = slot_size
    self.mlp_hidden_size = 2 * slot_size
    self.temp = temp

    self.norm_inputs = layers.LayerNormalization()
    self.norm_slots = layers.LayerNormalization()
    self.norm_mlp = layers.LayerNormalization()

    # # Parameters for Gaussian init (shared by all slots).
    if learn_initial_dist:
      self.slots_mu = self.add_weight(
          initializer="glorot_uniform",
          shape=[1, 1, self.slot_size],
          dtype=tf.float32,
          name="slots_mu")
      self.slots_log_sigma = self.add_weight(
          initializer="glorot_uniform",
          shape=[1, 1, self.slot_size],
          dtype=tf.float32,
          name="slots_log_sigma")
    else:
      self.slots_mu = tf.constant(
          # initializer="glorot_uniform",
          tf.zeros([1, 1, self.slot_size]),
          dtype=tf.float32,
          name="slots_mu")
      self.slots_log_sigma = tf.constant(
          # initializer="glorot_uniform",
          tf.zeros([1, 1, self.slot_size]),
          dtype=tf.float32,
          name="slots_log_sigma")

    # Linear maps for the attention module.
    self.project_q = layers.Dense(self.slot_size, use_bias=False, name="q")
    self.project_k = layers.Dense(self.slot_size, use_bias=False, name="k")
    self.project_v = layers.Dense(self.slot_size, use_bias=False, name="v")

    # Slot update functions.
    self.gru = layers.GRUCell(self.slot_size)
    self.mlp = tf.keras.Sequential([
        layers.Dense(self.mlp_hidden_size, activation="relu"),
        layers.Dense(self.slot_size)
    ], name="mlp")

  def reset(self, batch_size):
    # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
    # slots_mu = tf.cast(self.slots_mu, tf.float32)
    # slots_log_sigma = tf.cast(self.slots_log_sigma, tf.float32)
    # noise = tf.random.normal([batch_size, self.num_slots, self.slot_size], dtype=tf.float32)

    # slots = slots_mu + tf.exp(slots_log_sigma) * noise
    slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal([batch_size, self.num_slots, self.slot_size])
    return slots

  def call(self, slots, inputs):
    # `inputs` has shape [batch_size, num_inputs, inputs_size].
    inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
    k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
    v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):
        slots_prev = slots
        slots = self.norm_slots(slots)

        # Attention.
        q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
        q *= (self.slot_size ** -0.5 / self.temp)  # Normalization.
        attn_logits = tf.keras.backend.batch_dot(k, q, axes=-1)
        attn = tf.nn.softmax(attn_logits, axis=-1)
        # `attn` has shape: [batch_size, num_inputs, num_slots].

        # Weigted mean.
        attn += self.epsilon
        attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
        updates = tf.keras.backend.batch_dot(attn, v, axes=-2)
        # `updates` has shape: [batch_size, num_slots, slot_size].

        # Slot update.
        slots, _ = self.gru(updates, [slots_prev])
        slots += self.mlp(self.norm_mlp(slots))

    return slots


def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = rearrange(slots, 'b k d -> (b k) d')
  grid = repeat(slots, f'bk d -> bk {resolution[0]} {resolution[1]} d')
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid


def spatial_flatten(x):
  return rearrange(x, 'b h w c -> b (h w) c')


def unstack_and_split(x, batch_size, num_channels=3):
  """Unstack batch dimension and split into channels and alpha mask."""
  unstacked = rearrange(x, '(b k) ... -> b k ...', b=batch_size)
  channels, masks = tf.split(unstacked, [num_channels, 1], axis=-1)
  return channels, masks

class DebugSlotAttentionEncoder(layers.Layer):
  def __init__(self, resolution, outdim):
    super().__init__()
    self.resolution = resolution
    self.outdim = outdim

    self.encoder_cnn = tf.keras.Sequential([
        layers.Conv2D(2, kernel_size=5, strides=2, activation="relu"),
        layers.Conv2D(2, kernel_size=3, activation="relu"),
        layers.Conv2D(2, kernel_size=3, activation="relu"),
        layers.Conv2D(4, kernel_size=3, activation="relu")
    ], name="encoder_cnn")
    self.encoder_pos = SoftPositionEmbed(4, self.resolution)
    self.layer_norm = layers.LayerNormalization()
    self.mlp = tf.keras.Sequential([
        layers.Dense(4, activation="relu"),
        layers.Dense(self.outdim)
    ], name="feedforward")

  def call(self, image):
    # `image` has shape: [batch_size, width, height, num_channels].

    # Convolutional encoder with position embedding.
    x = self.encoder_cnn(image)  # CNN Backbone.
    x = self.encoder_pos(x)  # Position embedding.
    x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
    x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
    # `x` has shape: [batch_size, width*height, input_size].
    return x

class SlimSlotAttentionEncoder(layers.Layer):
  def __init__(self, resolution, outdim):
    super().__init__()
    self.resolution = resolution
    self.outdim = outdim

    self.encoder_cnn = tf.keras.Sequential([
        layers.Conv2D(8, kernel_size=4, strides=2, activation="relu"),
        layers.Conv2D(16, kernel_size=4, padding="SAME", activation="relu"),
        layers.Conv2D(32, kernel_size=4, padding="SAME", activation="relu"),
    ], name="encoder_cnn")
    self.encoder_pos = SoftPositionEmbed(32, (31, 31))
    self.layer_norm = layers.LayerNormalization()
    self.mlp = tf.keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(self.outdim)
    ], name="feedforward")

  def call(self, image):
    # `image` has shape: [batch_size, width, height, num_channels].
    # Convolutional encoder with position embedding.
    x = self.encoder_cnn(image)  # CNN Backbone.
    x = self.encoder_pos(x)  # Position embedding.
    x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
    x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
    # `x` has shape: [batch_size, width*height, input_size].
    return x

class SlotAttentionEncoder(layers.Layer):
  def __init__(self, resolution, outdim):
    super().__init__()
    self.resolution = resolution
    self.outdim = outdim

    self.encoder_cnn = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu")
    ], name="encoder_cnn")
    self.encoder_pos = SoftPositionEmbed(64, self.resolution)
    self.layer_norm = layers.LayerNormalization()
    self.mlp = tf.keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(self.outdim)
    ], name="feedforward")

  def call(self, image):
    # `image` has shape: [batch_size, width, height, num_channels].

    # Convolutional encoder with position embedding.
    x = self.encoder_cnn(image)  # CNN Backbone.
    x = self.encoder_pos(x)  # Position embedding.
    x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
    x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
    # `x` has shape: [batch_size, width*height, input_size].
    return x

class DebugSlotAttentionDecoder6464(layers.Layer):
  def __init__(self, in_dim):
    super().__init__()
    self.decoder_initial_size = (8, 8)
    self.in_dim = in_dim

    self.decoder_pos = SoftPositionEmbed(self.in_dim, self.decoder_initial_size)
    self.decoder_cnn = tf.keras.Sequential([
        layers.Conv2DTranspose(
            2, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            2, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            2, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            2, 5, strides=(1, 1), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            4, 3, strides=(1, 1), padding="SAME", activation=None)
    ], name="decoder_cnn")

  def call(self, slots):
    # Spatial broadcast decoder.
    x = spatial_broadcast(slots, self.decoder_initial_size)
    # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
    x = self.decoder_pos(x)
    x = self.decoder_cnn(x)
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].
    return x


class SlotAttentionDecoder(layers.Layer):
  def __init__(self, in_dim, resolution):
    super().__init__()
    self.decoder_initial_size = (8, 8)
    self.in_dim = in_dim
    self.resolution = resolution

    if self.resolution == (128, 128):
      num_layers = 4
    elif self.resolution == (64, 64):
      num_layers = 3
    else:
      raise NotImplementedError

    self.decoder_pos = SoftPositionEmbed(self.in_dim, self.decoder_initial_size)
    decoder_layers = [layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu")
      for _ in range(num_layers)]
    decoder_layers.extend([
        layers.Conv2DTranspose(
            64, 5, strides=(1, 1), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            4, 3, strides=(1, 1), padding="SAME", activation=None),
      ])
    self.decoder_cnn = tf.keras.Sequential(decoder_layers, name="decoder_cnn")

  def call(self, slots):
    # Spatial broadcast decoder.
    x = spatial_broadcast(slots, self.decoder_initial_size)
    # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
    x = self.decoder_pos(x)
    x = self.decoder_cnn(x)
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].
    return x

class SlimSlotAttentionDecoder(layers.Layer):
  def __init__(self, in_dim, resolution):
    super().__init__()
    self.decoder_initial_size = (8, 8)
    self.in_dim = in_dim
    self.resolution = resolution
    assert self.resolution == (64, 64)

    self.decoder_pos = SoftPositionEmbed(self.in_dim, self.decoder_initial_size)
    self.decoder_cnn = tf.keras.Sequential([
        layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            32, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            16, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            8, 5, strides=(1, 1), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            4, 3, strides=(1, 1), padding="SAME", activation=None)
    ], name="decoder_cnn")

  def call(self, slots):
    # Spatial broadcast decoder.
    x = spatial_broadcast(slots, self.decoder_initial_size)
    # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
    x = self.decoder_pos(x)
    x = self.decoder_cnn(x)
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].
    return x

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(layers.Layer):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.

    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super().__init__()
    self.dense = layers.Dense(hidden_size, use_bias=True)
    self.grid = build_grid(resolution)

  def call(self, inputs):
    return inputs + self.dense(self.grid)



