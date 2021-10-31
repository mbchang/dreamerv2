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
import ml_collections
from ml_collections.config_flags import config_flags
import tensorflow as tf

import datetime
import einops
import h5py
from loguru import logger as lgr
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import wandb

# import slot_attention.data as data_utils
import slot_attention_learners as model_utils
import slot_attention_utils as utils
# import balls_dataloader

launch_config = ml_collections.ConfigDict(dict(
  dataroot='ball_data/U-Dk4s5n5t10_ab',
  model_type='factorized_world_model',

  jobtype='train',
  seed=0,

  subroot='runs',
  expname='',

  monitoring=ml_collections.ConfigDict(dict(
      log_every=100,
      save_every=1000,
      vis_every=1000,
    )),

  system=ml_collections.ConfigDict(dict(
    cpu=False,
    headless=True))
  ))


FLAGS = flags.FLAGS

learners = dict(

  )


# then we'd have another config that decide which learner to use

config_flags.DEFINE_config_dict('lnch', launch_config)

config_flags.DEFINE_config_dict('lnr', model_utils.FactorizedWorldModel.get_default_args())

# what I want: give the model_type, and then it will load up the config dict for that model type and parse the args according to that 



class WhiteBallDataLoader():
  def __init__(self, h5):
    self.h5 = h5
    assert 'observations' in self.h5.keys() and 'actions' in self.h5.keys()

  def normalize_actions(self, act_batch):
    # normalize actions from [0, 5] to [-1, 1]
    act_batch = (act_batch * 2./5) - 1
    return act_batch

  def get_batch(self, batch_size, num_frames):
    batch_indices = np.random.choice(self.h5['observations'].shape[0], size=batch_size, replace=False)
    obs_batch = self.h5['observations'][sorted(batch_indices), :num_frames]
    obs_batch = utils.normalize(obs_batch)
    obs_batch = einops.rearrange(obs_batch, '... c h w -> ... h w c')
    obs_batch = tf.convert_to_tensor(obs_batch)
    if num_frames > 1:
      act_batch = self.h5['actions'][sorted(batch_indices), :num_frames]
      act_batch = act_batch[:, 1:]  # (B, T-1, A)
      act_batch = self.normalize_actions(act_batch)
      act_batch = tf.convert_to_tensor(act_batch)
      return {'image': obs_batch, 'action': act_batch}
    else:
      obs_batch = einops.rearrange(obs_batch, 'b t ... -> (b t) ...')
      return {'image': obs_batch}


def main(argv):
  # print(argv)

  args = ml_collections.ConfigDict(FLAGS.lnch.to_dict())


  # config_flags.DEFINE_config_dict('lnr', model_utils.FactorizedWorldModel.get_default_args())

  lnr_args = ml_collections.ConfigDict(FLAGS.lnr.to_dict())

  tf.random.set_seed(args.seed)
  np.random.seed(args.seed)
  resolution = (64, 64)

  if args.model_type == 'object_discovery':
    assert lnr_args.num_frames == 1

  tf.config.run_functions_eagerly(True)
  if not args.system.cpu:
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

  expdir = os.path.join(args.subroot, args.expname)
  os.makedirs(expdir, exist_ok=True)

  wandb.init(
      config=lnr_args.to_dict(),
      project='slot attention',
      dir=expdir,
      group=os.path.basename(args.subroot),
      job_type=args.jobtype,
      id=args.expname+'_{date:%Y%m%d%H%M%S}'.format(
          date=datetime.datetime.now())
      )

  lgr.remove()   # remove default handler
  lgr.add(os.path.join(expdir, 'debug.log'))
  if not args.system.headless:
    lgr.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

  # Build dataset iterators, optimizers and model.
  # data_iterator = data_utils.build_clevr_iterator(
  #     batch_size, split="train", resolution=resolution, shuffle=True,
  #     max_n_objects=6, get_properties=False, apply_crop=True)
  data_iterator = WhiteBallDataLoader(h5=h5py.File(f'{args.dataroot}.h5', 'r'))

  optimizer = tf.keras.optimizers.Adam(lnr_args.optim.learning_rate, epsilon=1e-08)

  model = model_utils.build_model(resolution, lnr_args.optim.batch_size, lnr_args.sess.num_slots, lnr_args.model.temp, model_type=args.model_type)

  # Prepare checkpoint manager.
  global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)
  ckpt = tf.train.Checkpoint(
      network=model, optimizer=optimizer, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=expdir, max_to_keep=5)
  ckpt.restore(ckpt_manager.latest_checkpoint)
  if ckpt_manager.latest_checkpoint:
    lgr.info(f"Restored from {ckpt_manager.latest_checkpoint}")
  else:
    lgr.info("Initializing from scratch.")

  start = time.time()
  for itr in range(lnr_args.optim.num_train_steps):
    # batch = next(data_iterator)
    batch = data_iterator.get_batch(lnr_args.optim.batch_size, lnr_args.sess.num_frames)

    # Learning rate warm-up.
    # if global_step < lnr_args.warmup_steps:
    #   learning_rate = lnr_args.learning_rate * tf.cast(
    #       global_step, tf.float32) / tf.cast(lnr_args.warmup_steps, tf.float32)
    # else:
    #   learning_rate = lnr_args.learning_rate
    # learning_rate = learning_rate * (lnr_args.decay_rate ** (
    #     tf.cast(global_step, tf.float32) / tf.cast(lnr_args.decay_steps, tf.float32)))

    if global_step < lnr_args.optim.warmup_steps:
      learning_rate = tf.cast(lnr_args.optim.learning_rate, tf.float32)
    else:
      learning_rate = lnr_args.optim.learning_rate * (lnr_args.optim.decay_rate ** (
          tf.cast(global_step-lnr_args.optim.warmup_steps, tf.float32) / tf.cast(lnr_args.optim.decay_steps, tf.float32)))

    # learning_rate = tf.cast(lnr_args.learning_rate, tf.float32)

    optimizer.lr = learning_rate.numpy()

    loss_value, _, _ = model.train_step(batch=batch, optimizer=optimizer)

    # Update the global step. We update it before logging the loss and saving
    # the model so that the last checkpoint is saved at the last iteration.
    global_step.assign_add(1)

    # Log the training loss.
    if not global_step % args.monitoring.log_every:
      lgr.info(f"Step: {global_step.numpy()}, Loss: {loss_value:.6f}, LR: {learning_rate.numpy():.3e}, Time: {datetime.timedelta(seconds=time.time() - start)}")

      wandb.log({
          f'{args.jobtype}/itr': global_step.numpy(),
          f'{args.jobtype}/loss': loss_value,
          f'{args.jobtype}/learning_rate': learning_rate.numpy(),
          }, step=global_step.numpy())

    # We save the checkpoints every 1000 iterations.
    if not global_step  % args.monitoring.save_every:
      # Save the checkpoint of the model.
      saved_ckpt = ckpt_manager.save()
      lgr.info(f"Saved checkpoint: {saved_ckpt}")

    if not global_step % args.monitoring.vis_every:
      if lnr_args.sess.num_frames > 1:
        # sequence_length = lnr_args.num_frames
        sequence_length = 10
        assert lnr_args.sess.pred_horizon < sequence_length
        seed_steps = sequence_length-lnr_args.sess.pred_horizon

        batch = data_iterator.get_batch(lnr_args.optim.batch_size, sequence_length)
        video = model.visualize(batch, seed_steps=seed_steps, pred_horizon=lnr_args.sess.pred_horizon)  # for now
        utils.save_gif(utils.add_border(video.numpy(), seed_steps), os.path.join(expdir, f'{global_step.numpy()}'))
      else:
        model.visualize(os.path.join(expdir, f'{global_step.numpy()}'), batch)


if __name__ == "__main__":
  # import sys
  # print(sys.argv)

  # maybe I can decide which configs to invoke here? 

  app.run(main)

"""
  # python train_slot_attention.py --batch_size 3 --subroot runs/sanity
  # CUDA_VISIBLE_DEVICES=0 python train_slot_attention.py --dataroot ball_data/Dk4s0n2000t10_ab --model_dir runs/sanity_again 


  10/24/21
  CUDA_VISIBLE_DEVICES=2 python train_slot_attention.py --dataroot ball_data/Dk4s0n2000t10_ab --expname t1_b16


  python train_slot_attention.py --batch_size 3 --subroot runs/sanity --cpu --headless=False --log_every 1 --lnr_args.num_train_steps 10


  10/25/21
  CUDA_VISIBLE_DEVICES=0 python train_slot_attention.py --dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --expname t3_b32 --model_type factorized_world_model --num_frames 3 --batch_size 32

  debug:
  python train_slot_attention.py --batch_size 2 --subroot runs/sanity --cpu --headless=False --log_every 1 --lnr_args.num_train_steps 5 --model_type factorized_world_model --num_frames 3 --vis_every 1

  2:54pm
  CUDA_VISIBLE_DEVICES=1 python train_slot_attention.py --dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --expname t3_ph1_b32_geb --model_type factorized_world_model --num_frames 3 --batch_size 32 --pred_horizon 1 &

  CUDA_VISIBLE_DEVICES=0 python train_slot_attention.py --dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --expname t4_ph2_b32_geb --model_type factorized_world_model --num_frames 4 --batch_size 32 --pred_horizon 2 &


  10/17/21
  debug:
    python train_slot_attention.py --batch_size 3 --subroot runs/sanity --cpu --headless=False --log_every 1 --lnr_args.num_train_steps 10
    python train_slot_attention.py --batch_size 2 --subroot runs/sanity --cpu --headless=False --log_every 1 --lnr_args.num_train_steps 5 --model_type factorized_world_model --num_frames 3 --vis_every 1 --pred_horizon 1

10/30/21
python train_slot_attention_mlc.py --lnr.optim.batch_size 2 --lnch.subroot runs/sanity --lnch.system.cpu --lnch.system.headless=False --lnch.monitoring.log_every 1 --lnr.optim.num_train_steps 5 --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnch.monitoring.vis_every 1 --lnr.sess.pred_horizon 1


"""

