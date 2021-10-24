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

"""Training loop for object discovery with Slot Attention."""
import datetime
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import einops
import h5py
from loguru import logger as lgr
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

# import slot_attention.data as data_utils
import slot_attention as model_utils
import slot_attention_utils as utils
# import balls_dataloader


FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "/tmp/object_discovery/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 5, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("num_train_steps", 500000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 10000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 100000,
                     "Number of steps for the learning rate decay.")

flags.DEFINE_string("dataroot", "ball_data/U-Dk4s5n5t10_ab", "path to h5 file")
flags.DEFINE_bool("cpu", False, "use cpu")
flags.DEFINE_bool("headless", True, "headless")


class WhiteBallDataLoader():
  def __init__(self, h5):
    self.h5 = h5
    assert 'observations' in self.h5.keys() and 'actions' in self.h5.keys()

  def get_batch(self, batch_size, num_frames):
    batch_indices = np.random.choice(self.h5['observations'].shape[0], size=batch_size, replace=False)
    obs_batch = self.h5['observations'][sorted(batch_indices), :num_frames]
    obs_batch = normalize(obs_batch)
    obs_batch = einops.rearrange(obs_batch, '... c h w -> ... h w c')
    obs_batch = tf.convert_to_tensor(obs_batch)
    if num_frames > 1:
      act_batch = self.h5['actions'][sorted(batch_indices), :num_frames]
      act_batch = act_batch[1:]  # (T-1, B, A)
      act_batch = tf.convert_to_tensor(act_batch)
      return {'image': obs_batch, 'action': act_batch}
    else:
      obs_batch = einops.rearrange(obs_batch, 'b t ... -> (b t) ...')
      return {'image': obs_batch}


# We use `tf.function` compilation to speed up execution. For debugging,
# consider commenting out the `@tf.function` decorator.
@tf.function
def train_step(batch, model, optimizer):
  """Perform a single training step."""

  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    preds = model(batch["image"], training=True)
    recon_combined, recons, masks, slots = preds
    loss_value = utils.l2_loss(batch["image"], recon_combined)
    del recons, masks, slots  # Unused.

  # Get and apply gradients.
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))

  return loss_value

def normalize(x):
  return (x - 0.5) * 2.0  # Rescale to [-1, 1]

def renormalize(x):
  """Renormalize from [-1, 1] to [0, 1]."""
  return x / 2. + 0.5

def get_prediction(model, batch, idx=0):
  recon_combined, recons, masks, _ = model(batch["image"])
  image = renormalize(batch["image"])[idx]
  recon_combined = renormalize(recon_combined)[idx]
  recons = renormalize(recons)[idx]
  masks = masks[idx]
  return image, recon_combined, recons, masks

def visualize(itr, batch, model):
  image, recon_combined, recons, masks = get_prediction(model, batch)
  # print(np.max(image), np.min(image))
  # print(np.max(recon_combined), np.min(recon_combined))
  # print(np.max(recons), np.min(recons))
  # print(np.max(masks), np.min(masks))
  # # assert False
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
  plt.savefig(os.path.join(FLAGS.model_dir, f'{itr}.png'))
  plt.close()

def main(argv):
  del argv
  # Hyperparameters of the model.
  batch_size = FLAGS.batch_size
  num_slots = FLAGS.num_slots
  num_iterations = FLAGS.num_iterations
  base_learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  warmup_steps = FLAGS.warmup_steps
  decay_rate = FLAGS.decay_rate
  decay_steps = FLAGS.decay_steps
  tf.random.set_seed(FLAGS.seed)
  resolution = (64, 64)

  tf.config.experimental_run_functions_eagerly(True)
  if not FLAGS.cpu:
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

  os.makedirs(FLAGS.model_dir, exist_ok=True)

  lgr.remove()   # remove default handler
  lgr.add(os.path.join(FLAGS.model_dir, 'debug.log'))
  if not FLAGS.headless:
    lgr.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

  # Build dataset iterators, optimizers and model.
  # data_iterator = data_utils.build_clevr_iterator(
  #     batch_size, split="train", resolution=resolution, shuffle=True,
  #     max_n_objects=6, get_properties=False, apply_crop=True)
  data_iterator = WhiteBallDataLoader(h5=h5py.File(f'{FLAGS.dataroot}.h5', 'r'))

  optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)

  model = model_utils.build_model(resolution, batch_size, num_slots,
                                  num_iterations, model_type="object_discovery")

  # Prepare checkpoint manager.
  global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)
  ckpt = tf.train.Checkpoint(
      network=model, optimizer=optimizer, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=5)
  ckpt.restore(ckpt_manager.latest_checkpoint)
  if ckpt_manager.latest_checkpoint:
    lgr.info(f"Restored from {ckpt_manager.latest_checkpoint}")
  else:
    lgr.info("Initializing from scratch.")

  start = time.time()
  for itr in range(num_train_steps):
    # batch = next(data_iterator)
    batch = data_iterator.get_batch(batch_size, 1)

    # Learning rate warm-up.
    if global_step < warmup_steps:
      learning_rate = base_learning_rate * tf.cast(
          global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    else:
      learning_rate = base_learning_rate
    learning_rate = learning_rate * (decay_rate ** (
        tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
    optimizer.lr = learning_rate.numpy()

    loss_value = train_step(batch, model, optimizer)

    # Update the global step. We update it before logging the loss and saving
    # the model so that the last checkpoint is saved at the last iteration.
    global_step.assign_add(1)

    # Log the training loss.
    if not global_step % 100:
      # lgr.info("Step: %s, Loss: %.6f, Time: %s".format(
      #              global_step.numpy(), loss_value,
      #              datetime.timedelta(seconds=time.time() - start)))
      lgr.info(f"Step: {global_step.numpy()}, Loss: {loss_value:.6f}, Time: {datetime.timedelta(seconds=time.time() - start)}")

    # We save the checkpoints every 1000 iterations.
    if not global_step  % 1000:
      # Save the checkpoint of the model.
      saved_ckpt = ckpt_manager.save()
      lgr.info(f"Saved checkpoint: {saved_ckpt}")

    if not global_step % 1000:
      visualize(global_step.numpy(), batch, model)


if __name__ == "__main__":
  app.run(main)

  # python train_slot_attention.py --batch_size 3 --model_dir runs/sanity
  # python train_slot_attention.py --dataroot ball_data/Dk4s0n2000t10_a --model_dir runs/sanity
