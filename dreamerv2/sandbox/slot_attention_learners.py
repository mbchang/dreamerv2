from einops import rearrange, repeat
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

import slot_attention as sa
import slot_attention_utils as utils
import attention


class SlotAttentionAutoEncoder(layers.Layer):
  """Slot Attention-based auto-encoder for object discovery."""

  """
config = ml_collections.ConfigDict(
  dict(
    subroot='runs',
    expname='',
    seed=0,
    batch_size=64,
    num_slots=5,
    num_frames=1,
    learning_rate=0.0004,
    num_train_steps=500000,
    warmup_steps=10000,
    decay_rate=0.5,
    decay_steps=100000,
    dataroot='ball_data/U-Dk4s5n5t10_ab',
    cpu=False,
    headless=True,
    jobtype='train',
    model_type='factorized_world_model',
    pred_horizon=0,
    slot_temp=1,
    ))


monitoring_config = ml_collections.ConfigDict(dict(
      log_every=100,
      save_every=1000,
      vis_every=1000,
      ))
  """

  def __init__(self, resolution, num_slots, temp):
    """Builds the Slot Attention-based auto-encoder.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
    """
    super().__init__()
    self.resolution = resolution
    self.num_slots = num_slots

    self.encoder = sa.SlotAttentionEncoder(self.resolution, 64)
    self.slot_attention = sa.SlotAttention(slot_size=64, temp=temp)
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

    recon_combined, comp, masks = self.decode(slots)
    return recon_combined, comp, masks, slots

  def decode(self, slots):
    # Spatial broadcast decoder.
    x = self.decoder(slots)
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

    # Undo combination of slot and batch dimension; split alpha masks.
    comp, masks = sa.unstack_and_split(x, batch_size=slots.shape[0])
    # `comp` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    masks = tf.nn.softmax(masks, axis=1)
    recon_combined = tf.reduce_sum(comp * masks, axis=1)  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].
    return recon_combined, comp, masks


  # We use `tf.function` compilation to speed up execution. For debugging,
  # consider commenting out the `@tf.function` decorator.
  @tf.function
  def train_step(self, batch, optimizer):
    """Perform a single training step."""

    # Get the prediction of the models and compute the loss.
    with tf.GradientTape() as tape:
      preds = self(batch["image"], training=True)
      recon_combined, comp, masks, slots = preds
      loss_value = utils.l2_loss(batch["image"], recon_combined)
      del comp, masks, slots  # Unused.

    # Get and apply gradients.
    gradients = tape.gradient(loss_value, self.trainable_weights)
    optimizer.apply_gradients(zip(gradients, self.trainable_weights))

    # dummy variables
    output = None
    metrics = None

    return loss_value, output, metrics

  def visualize(self, fname, batch, idx=0):
    recon_combined, comp, masks, _ = self(batch["image"])
    image = utils.renormalize(batch["image"])[idx]
    recon_combined = utils.renormalize(recon_combined)[idx]
    comp = utils.renormalize(comp)[idx]
    masks = masks[idx]

    num_slots = len(masks)
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Recon.')
    for i in range(num_slots):
      ax[i + 2].imshow(comp[i] * masks[i] + (1 - masks[i]))
      ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
      ax[i].grid(False)
      ax[i].axis('off')
    plt.savefig(f'{fname}.png')
    plt.close()


class FactorizedWorldModel(layers.Layer):
  # model specific args
  # video_pred: {seed_steps: 3, prediction_horizon: 7, num_ex: 5}


  """
  Good hyperparameters
    - batch_size 32 (although I don't think this makes that much of a diff)
    - lr: 1e-4 --> this seems to be important
    - warmup_steps: 0 (although maybe having some warmup will help?)
    - decay_rate: 0.5 (although I have not really tried others --> I should probably decay this more aggressively, or reduce the number of decay steps)
    - slot_temp: 0.5 --> this seems to be important
  """

  @staticmethod
  def get_default_args():
      default_args = ml_collections.ConfigDict(
        dict(
          optim=ml_collections.ConfigDict(dict(
            batch_size=32,
            decay_rate=0.5,
            decay_steps=25000,
            learning_rate=1e-4,
            num_train_steps=500000,
            warmup_steps=10000,
            )),
          model=ml_collections.ConfigDict(dict(
            resolution=(64, 64),
            temp=0.5,
            encoder_type='slim',
            decoder_type='slim',
            posterior_loss=False,
            overshooting_loss=True,
            )),
          sess=ml_collections.ConfigDict(dict(
            num_slots=5,
            num_frames=3,
            pred_horizon=7,
            )),
        # for now
        monitoring=ml_collections.ConfigDict(dict(
          log_every=100,
          save_every=1000,
          vis_every=1000))
          ))
      return default_args


  def __init__(self, resolution, num_slots, temp, 
    # later you will just pass the config in
    encoder_type='slim', 
    decoder_type='slim',
    posterior_loss=False,
    overshooting_loss=True,
    ):
    """Builds the Slot Attention-based auto-encoder.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
    """
    super().__init__()
    self.resolution = resolution
    self.num_slots = num_slots

    # replace this with config at some point
    self.posterior_loss = posterior_loss
    self.overshooting_loss = overshooting_loss

    if encoder_type == 'default':
      self.encoder = sa.SlotAttentionEncoder(self.resolution, 64)
    elif encoder_type == 'slim':
      self.encoder = sa.SlimSlotAttentionEncoder(self.resolution, 64)
    else:
      raise NotImplementedError

    self.slot_attention = sa.SlotAttention(slot_size=64, temp=temp)
    self.slot_attention.register_num_slots(self.num_slots)

    if decoder_type == 'default':
      self.decoder = sa.SlotAttentionDecoder(64, resolution)
    elif decoder_type == 'slim':
      self.decoder = sa.SlimSlotAttentionDecoder(64, resolution)
    else:
      raise NotImplementedError

    self.action_encoder = tf.keras.Sequential([
      layers.Dense(64, activation='relu'),
      layers.Dense(64)
      ])
    self.dynamics = attention.CrossAttentionBlock(dim=64, num_heads=1)

  def call(self, data):
    """
      posterior[0] corresponds to the prediction for data[0]
      prior[0] corresponds to the posterior for data[1]

      data['image']: (B, T, H, W, C)
      data['action']: (B, T-1, A)

      print(output['prior']['latent'].shape)  # (2, 3, 5, 64)
      print(output['prior']['pred']['comb'].shape)  # (2, 3, 64, 64, 3)
      print(output['prior']['pred']['comp'].shape)  # (2, 3, 5, 64, 64, 3)
      print(output['prior']['pred']['masks'].shape)  # (2, 3, 5, 64, 64, 1)
      print(output['posterior']['latent'].shape)  # (2, 4, 5, 64)
      print(output['posterior']['pred']['comb'].shape)  # (2, 4, 64, 64, 3)
      print(output['posterior']['pred']['comp'].shape)  # (2, 4, 5, 64, 64, 3)
      print(output['posterior']['pred']['masks'].shape)  # (2, 4, 5, 64, 64, 1)
    """ 
    # print(f"image | shape: {data['image'].shape} min: {tf.reduce_min(data['image'])} max: {tf.reduce_max(data['image'])}")
    # print(f"action | shape: {data['action'].shape} min: {tf.reduce_min(data['action'])} max: {tf.reduce_max(data['action'])}")

    bsize = data['image'].shape[0]
    embed = utils.bottle(self.encoder)(data['image'])  # (b, t, h*w, do)
    priors, posteriors = self.filter(
      slots=self.slot_attention.reset(batch_size=bsize),  # (b, k, ds)
      embeds=embed, 
      actions=data['action'])
    prior_comb, prior_comp, prior_masks = utils.bottle(self.decode)(priors)
    post_comb, post_comp, post_masks = utils.bottle(self.decode)(posteriors)
    output = dict(
      prior=dict(
        latent=priors,
        pred=dict(comb=prior_comb, comp=prior_comp, masks=prior_masks)),
      posterior=dict(
        latent = posteriors,
        pred=dict(comb=post_comb, comp=post_comp, masks=post_masks)))
    return output

  def decode(self, slots):
    # Spatial broadcast decoder.
    x = self.decoder(slots)
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

    # Undo combination of slot and batch dimension; split alpha masks.
    comp, masks = sa.unstack_and_split(x, batch_size=slots.shape[0])
    # `comp` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    masks = tf.nn.softmax(masks, axis=1)
    recon_combined = tf.reduce_sum(comp * masks, axis=1)  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].
    return recon_combined, comp, masks

  def filter(self, slots, embeds, actions):
    """
      slots: (b, k, ds)
      embeds: (b, t, h*w, do)
      actions: (b, t-1, da)

      this can be done using a transformer
      you can also try and see how the static scan thing works
    """
    bsize = slots.shape[0]
    actions = self.action_encoder(actions)

    priors, posteriors = [], []
    # initial step
    posterior = self.slot_attention(slots, embeds[:, 0])
    posteriors.append(posterior)
    # subsequent steps
    for i in range(actions.shape[1]):
      context = tf.concat([posterior, rearrange(actions[:, i], 'b a -> b 1 a')], 1)
      # prior: t-1 to t'
      prior = self.dynamics(posterior, context)
      # posterior t' to t
      posterior = self.slot_attention(prior, embeds[:, i+1])
      priors.append(prior)
      posteriors.append(posterior)

    priors = rearrange(priors, 't b ... -> b t ...')
    posteriors = rearrange(posteriors, 't b ... -> b t ...')
    return priors, posteriors

  def generate(self, slots, actions):
    """
      slots: (b, k, ds)
      actions: (b, t, da)
    """
    bsize = slots.shape[0]
    actions = self.action_encoder(actions)

    latents = []
    for i in range(actions.shape[1]):
      context = tf.concat([slots, rearrange(actions[:, i], 'b a -> b 1 a')], 1)
      slots = self.dynamics(slots, context)
      latents.append(slots)

    latents = rearrange(latents, 't b ... -> b t ...')
    return latents

  @tf.function
  def train_step(self, batch, optimizer):
    with tf.GradientTape() as tape:
      output = self(batch, training=True)

      metrics = {}
      metrics['prior'] = utils.l2_loss(batch['image'][:, 1:], output['prior']['pred']['comb'])

      dtype = metrics['prior'].dtype

      if self.posterior_loss:
        metrics['posterior'] = utils.l2_loss(batch['image'], output['posterior']['pred']['comb'])
      else:
        metrics['posterior'] = tf.cast(tf.convert_to_tensor(0), dtype)

      # add overshooting loss here? 
      # self.overshooting_loss = True
      if self.overshooting_loss:
        metrics['overshoot'] = tf.cast(tf.convert_to_tensor(0), dtype)
        for i in range(1, batch['action'].shape[1]):
          # print(f"overshoot i={i} gt:{batch['image'][:, i+1:].shape} actions: {batch['action'][:, i:].shape} latent: {output['posterior']['latent'][:, i].shape}")
          metrics['overshoot'] += utils.l2_loss(
            batch['image'][:, i+1:], 
            self.imagine(
              slots=output['posterior']['latent'][:, i],
              actions=batch['action'][:, i:])['pred']['comb'])
      else:
        metrics['overshoot'] = tf.cast(tf.convert_to_tensor(0), dtype)

      # latent loss
      metrics['subsequent_latent'] = utils.l2_loss(
        output['posterior']['latent'][:, 1:], output['prior']['latent'])
      metrics['initial_latent'] = tf.cast(tf.convert_to_tensor(0), dtype)  # do we want this?

      loss_value = tf.reduce_sum(list(metrics.values()))

    # Get and apply gradients.
    gradients = tape.gradient(loss_value, self.trainable_weights)
    optimizer.apply_gradients(zip(gradients, self.trainable_weights))

    return loss_value, output, metrics

  def imagine(self, slots, actions):
    """
      slots: (b, k, ds)
      actions: (b, t, da)
    """
    bsize = slots.shape[0]
    imag_latent = self.generate(slots, actions)
    imag_comb, imag_comp, imag_masks = utils.bottle(self.decode)(imag_latent)
    imag_output = dict(
      latent=imag_latent,
      pred=dict(comb=imag_comb, comp=imag_comp, masks=imag_masks)
      )
    return imag_output

  def visualize(self, batch, seed_steps=3, pred_horizon=5, num_ex=5):
    obs = batch['image'][:num_ex, :seed_steps + pred_horizon]
    act = batch['action'][:num_ex, :seed_steps - 1 + pred_horizon]
    recon_output = self({'image': obs[:, :seed_steps], 'action': act[:, :seed_steps-1]})
    imag_output = self.imagine(recon_output['posterior']['latent'][:, -1], act[:, seed_steps-1:])

    recon_comb = recon_output['posterior']['pred']['comb']
    recon_comp = recon_output['posterior']['pred']['comp']
    recon_masks = recon_output['posterior']['pred']['masks']
    imag_comb = imag_output['pred']['comb']
    imag_comp = imag_output['pred']['comp']
    imag_masks = imag_output['pred']['masks']

    truth = utils.renormalize(obs)  # (b, t, h, w, c)
    model = utils.renormalize(tf.concat([recon_comb, imag_comb], 1))  # (b, t, h, w, c)
    components = tf.concat([
      utils.renormalize(recon_comp) * recon_masks + (1 - recon_masks),
      utils.renormalize(imag_comp) * imag_masks + (1 - imag_masks),
      ], 1)  # (b, t, k, h, w, c)
    error = (model - truth + 1) / 2  # (b, t, h, w, c)

    video = tf.concat([truth, model, error, rearrange(components, 'b t k h w c -> b t (k h) w c')], 2)
    video = rearrange(video, 'b t h w c -> t h (b w) c')
    return video


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


def build_model(resolution, batch_size, num_slots, temp, model_type="object_discovery"):
  num_channels = 3

  """Build keras model."""
  if model_type == "object_discovery":
    model = SlotAttentionAutoEncoder(resolution, num_slots, temp)
    # image = tf.keras.Input(list(resolution) + [num_channels], batch_size)
    # outputs = model(image)
    # model = tf.keras.Model(inputs=image, outputs=outputs)
  elif model_type == "set_prediction":
    model = SlotAttentionClassifier(resolution, num_slots)
    # image = tf.keras.Input(list(resolution) + [num_channels], batch_size)
    # outputs = model(image)
    # model = tf.keras.Model(inputs=image, outputs=outputs)
  elif model_type == 'factorized_world_model':
    model = FactorizedWorldModel(resolution, num_slots, temp)
    # image = tf.keras.Input([1] + list(resolution) + [num_channels], batch_size)
    # outputs = model(image)
    # model = tf.keras.Model(inputs=image, outputs=outputs)
  else:
    raise ValueError("Invalid name for model type.")

  return model

def get_learner(model_type):
  learners = {
    'object_discovery': SlotAttentionAutoEncoder,
    'set_prediction': SlotAttentionClassifier,
    'factorized_world_model': FactorizedWorldModel
  }
  return learners[model_type]


