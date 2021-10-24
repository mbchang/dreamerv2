from einops import rearrange, repeat
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

import slot_attention as sa
import slot_attention_utils as utils


class SlotAttentionAutoEncoder(layers.Layer):
  """Slot Attention-based auto-encoder for object discovery."""

  def __init__(self, resolution, num_slots, num_iterations):
    """Builds the Slot Attention-based auto-encoder.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
      num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.resolution = resolution
    self.num_slots = num_slots
    self.num_iterations = num_iterations

    self.encoder = sa.SlotAttentionEncoder(self.resolution, 64)
    self.slot_attention = sa.SlotAttention(
        num_iterations=self.num_iterations,
        # num_slots=self.num_slots,
        slot_size=64,
        mlp_hidden_size=128)
    self.slot_attention.register_num_slots(self.num_slots)
    self.decoder = sa.SlotAttentionDecoder(64, resolution)

  def call(self, image):
    # `image` has shape: [batch_size, width, height, num_channels].

    # # Convolutional encoder with position embedding.
    x = self.encoder(image)
    # `x` has shape: [batch_size, width*height, input_size].

    # Slot Attention module.
    slots = self.slot_attention.reset(batch_size=tf.shape(x)[0])
    slots = self.slot_attention(slots, x)
    # `slots` has shape: [batch_size, num_slots, slot_size].

    # Spatial broadcast decoder.
    x = self.decoder(slots)
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

    # Undo combination of slot and batch dimension; split alpha masks.
    recons, masks = sa.unstack_and_split(x, batch_size=image.shape[0])
    # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    masks = tf.nn.softmax(masks, axis=1)
    recon_combined = tf.reduce_sum(recons * masks, axis=1)  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].

    return recon_combined, recons, masks, slots


  # We use `tf.function` compilation to speed up execution. For debugging,
  # consider commenting out the `@tf.function` decorator.
  @tf.function
  def train_step(self, batch, optimizer):
    """Perform a single training step."""

    # Get the prediction of the models and compute the loss.
    with tf.GradientTape() as tape:
      preds = self(batch["image"], training=True)
      recon_combined, recons, masks, slots = preds
      loss_value = utils.l2_loss(batch["image"], recon_combined)
      del recons, masks, slots  # Unused.

    # Get and apply gradients.
    gradients = tape.gradient(loss_value, self.trainable_weights)
    optimizer.apply_gradients(zip(gradients, self.trainable_weights))

    return loss_value

  def get_prediction(self, batch, idx=0):
    recon_combined, recons, masks, _ = self(batch["image"])
    image = utils.renormalize(batch["image"])[idx]
    recon_combined = utils.renormalize(recon_combined)[idx]
    recons = utils.renormalize(recons)[idx]
    masks = masks[idx]
    return image, recon_combined, recons, masks

  def visualize(self, fname, batch):
    image, recon_combined, recons, masks = self.get_prediction(batch)
    num_slots = len(masks)
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Recon.')
    for i in range(num_slots):
      ax[i + 2].imshow(recons[i] * masks[i] + (1 - masks[i]))
      ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
      ax[i].grid(False)
      ax[i].axis('off')
    plt.savefig(fname)
    plt.close()



class SlotAttentionClassifier(layers.Layer):
  """Slot Attention-based classifier for property prediction."""

  def __init__(self, resolution, num_slots, num_iterations):
    """Builds the Slot Attention-based classifier.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
      num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.resolution = resolution
    self.num_slots = num_slots
    self.num_iterations = num_iterations

    self.encoder_cnn = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, strides=(2, 2),
                      padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, strides=(2, 2),
                      padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu")
    ], name="encoder_cnn")

    self.encoder_pos = sa.SoftPositionEmbed(64, (32, 32))

    self.layer_norm = layers.LayerNormalization()
    self.mlp = tf.keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64)
    ], name="feedforward")

    self.slot_attention = sa.SlotAttention(
        num_iterations=self.num_iterations,
        num_slots=self.num_slots,
        slot_size=64,
        mlp_hidden_size=128)

    self.mlp_classifier = tf.keras.Sequential(
        [layers.Dense(64, activation="relu"),
         layers.Dense(19, activation="sigmoid")],  # Number of targets in CLEVR.
        name="mlp_classifier")

  def call(self, image):
    # `image` has shape: [batch_size, width, height, num_channels].

    # Convolutional encoder with position embedding.
    x = self.encoder_cnn(image)  # CNN Backbone.
    x = self.encoder_pos(x)  # Position embedding.
    x = sa.spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
    x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
    # `x` has shape: [batch_size, width*height, input_size].

    # Slot Attention module.
    slots = self.slot_attention(x)
    # `slots` has shape: [batch_size, num_slots, slot_size].

    # Apply classifier per slot. The predictions have shape
    # [batch_size, num_slots, set_dimension].

    predictions = self.mlp_classifier(slots)

    return predictions


# def build_model(resolution, batch_size, num_slots, num_iterations,
#                 num_channels=3, model_type="object_discovery"):
#   """Build keras model."""
#   if model_type == "object_discovery":
#     model_def = SlotAttentionAutoEncoder
#   elif model_type == "set_prediction":
#     model_def = SlotAttentionClassifier
#   else:
#     raise ValueError("Invalid name for model type.")

#   image = tf.keras.Input(list(resolution) + [num_channels], batch_size)
#   outputs = model_def(resolution, num_slots, num_iterations)(image)
#   model = tf.keras.Model(inputs=image, outputs=outputs)
#   return model

def build_model(resolution, batch_size, num_slots, num_iterations,
                num_channels=3, model_type="object_discovery"):
  """Build keras model."""
  if model_type == "object_discovery":
    model_def = SlotAttentionAutoEncoder
  elif model_type == "set_prediction":
    model_def = SlotAttentionClassifier
  else:
    raise ValueError("Invalid name for model type.")

  return model_def(resolution, num_slots, num_iterations)