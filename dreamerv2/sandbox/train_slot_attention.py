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
import pathlib
import sys
import wandb

import slot_attention_learners as model_utils
import slot_attention_utils as utils

launch_config = ml_collections.ConfigDict(dict(
  dataroot='ball_data/U-Dk4s5n5t10_ab', # or 
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


class DMCDatLoader():
  def __init__(self, dataroot):
    self.episodes = DMCDatLoader.load_episodes(pathlib.Path(dataroot) / 'train_episodes')

  # taken from dreamer/replay
  @staticmethod
  def load_episodes(directory, capacity=None, minlen=1):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob('*.npz'))
    if capacity:
      num_steps = 0
      num_episodes = 0
      for filename in reversed(filenames):
        length = int(str(filename).split('-')[-1][:-4])
        num_steps += length
        num_episodes += 1
        if num_steps >= capacity:
          break
      filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
      try:
        with filename.open('rb') as f:
          episode = np.load(f)
          episode = {k: episode[k] for k in episode.keys()}
      except Exception as e:
        print(f'Could not load episode {str(filename)}: {e}')
        continue
      episodes[str(filename)] = episode
    return episodes

  def get_batch(self, batch_size, num_frames):
    obs_batch = []
    act_batch = []

    episodes = list(self.episodes.values())
    for i in range(batch_size):
      # sample random episode
      episode = episodes[np.random.randint(len(episodes))]
      total_length = len(episode['action'])
      assert total_length > num_frames

      # sample random chunk
      start = np.random.randint(total_length-num_frames+1)
      end = start + num_frames

      # append
      obs_batch.append(episode['image'][start:end])
      act_batch.append(episode['action'][start+1:end])  # drop first action

    # stack
    obs_batch = np.stack(obs_batch)
    act_batch = np.stack(act_batch)

    # normalize
    obs_batch = utils.normalize(obs_batch.astype(np.float32) / 255.0)
    batch =  {'image': obs_batch, 'action': act_batch}

    return batch


def main(argv):
  args = ml_collections.ConfigDict(FLAGS.lnch.to_dict())
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

  # if pathlib.Path(args.dataroot).parent.name == 'ball_data':
  if 'ball_data' in args.dataroot:
    data_iterator = WhiteBallDataLoader(h5=h5py.File(f'{args.dataroot}.h5', 'r'))
  # elif pathlib.Path(args.dataroot).parent.name == 'dmc_data':
  elif 'dmc_data' in args.dataroot:
    data_iterator = DMCDatLoader(dataroot=args.dataroot)
  else:
    raise NotImplementedError

  optimizer = tf.keras.optimizers.Adam(lnr_args.optim.learning_rate, epsilon=1e-08)

  # model = model_utils.build_model(resolution, lnr_args.optim.batch_size, lnr_args.sess.num_slots, lnr_args.model.temp, model_type=args.model_type)
  model = model_utils.get_learner(args.model_type)(num_slots=lnr_args.sess.num_slots, **lnr_args.model)

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

    if global_step < lnr_args.optim.warmup_steps:
    #   learning_rate = lnr_args.learning_rate * tf.cast(
    #       global_step, tf.float32) / tf.cast(lnr_args.warmup_steps, tf.float32)
      learning_rate = tf.cast(lnr_args.optim.learning_rate, tf.float32)
    else:
      learning_rate = lnr_args.optim.learning_rate * (lnr_args.optim.decay_rate ** (
          tf.cast(global_step-lnr_args.optim.warmup_steps, tf.float32) / tf.cast(lnr_args.optim.decay_steps, tf.float32)))

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
python train_slot_attention.py --lnr.optim.batch_size 2 --lnch.subroot runs/sanity --lnch.system.cpu --lnch.system.headless=False --lnch.monitoring.log_every 1 --lnr.optim.num_train_steps 5 --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnch.monitoring.vis_every 1 --lnr.sess.pred_horizon 1

10/31/21
3pm or so
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python train_slot_attention.py --dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --model_type factorized_world_model --num_frames 3 --batch_size 32 --pred_horizon 7 --learning_rate 0.0002 --decay_steps 25000 --slot_temp 0.5 --expname t3_ph7_b32_lr2e-4_dr5e-1_st5e-1_imagpost_ds25e3_wuconstant_normalized_geb_again &

slim encoder and decoder

6:22pm
[gauss1]

CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.learning_rate 0.0002 --lnr.optim.decay_steps 25000 --lnr.model.temp 0.5 --lnr.model.encoder_type slim --lnr.model.decoder_type slim --lnch.expname t3_ph7_b32_lr2e-4_dr5e-1_st5e-1_ds25e3_etslim_dtslim &

CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.learning_rate 0.0002 --lnr.optim.decay_steps 25000 --lnr.model.temp 0.5 --lnr.model.decoder_type slim --lnch.expname t3_ph7_b32_lr2e-4_dr5e-1_st5e-1_ds25e3_etdefault_dtslim &


CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.learning_rate 0.0002 --lnr.optim.decay_steps 25000 --lnr.model.temp 0.5 --lnr.model.encoder_type slim --lnch.expname t3_ph7_b32_lr2e-4_dr5e-1_st5e-1_ds25e3_etslim_dtdefault &

7:07pm
[geb] no posterior loss
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.learning_rate 0.0002 --lnr.optim.decay_steps 25000 --lnr.model.temp 0.5 --lnr.model.encoder_type slim --lnr.model.decoder_type slim --lnr.model.posterior_loss=False --lnch.expname t3_ph7_b32_lr2e-4_dr5e-1_st5e-1_ds25e3_etslim_dtslim_plFalse &

[geb] everything loss, including overshooting (t=10)
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 10 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.learning_rate 0.0002 --lnr.optim.decay_steps 25000 --lnr.model.temp 0.5 --lnr.model.encoder_type slim --lnr.model.decoder_type slim --lnr.model.posterior_loss=True --lnr.model.overshooting_loss=True --lnch.expname t10_ph7_b32_lr2e-4_dr5e-1_st5e-1_ds25e3_etslim_dtslim_plTrue_osTrue &

[gauss1] overshooting, no posterior
CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 8 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 5 --lnr.optim.learning_rate 0.0002 --lnr.optim.decay_steps 25000 --lnr.model.temp 0.5 --lnr.model.encoder_type slim --lnr.model.decoder_type slim --lnr.model.posterior_loss=False --lnr.model.overshooting_loss=True --lnch.expname t8_ph5_b32_lr2e-4_dr5e-1_st5e-1_ds25e3_etslim_dtslim_plFalse_osTrue &

[gauss1] slim, temp 1
CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.learning_rate 0.0002 --lnr.optim.decay_steps 25000 --lnr.model.temp 1.0 --lnr.model.encoder_type slim --lnr.model.decoder_type slim --lnch.expname t3_ph7_b32_lr2e-4_dr5e-1_st1_ds25e3_etslim_dtslim &


11/1/21
[geb] lr1e-4, st5e-1, slim, posterior=False
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.decay_steps 25000 --lnr.model.encoder_type slim --lnr.model.decoder_type slim --lnr.model.posterior_loss=False --lnr.model.overshooting_loss=False --lnch.expname 11_1_21_t3_ph7_b32_lr1e-4_dr5e-1_st5e-1_ds25e3_etslim_dtslim_plFalse_osFalse &

[gauss1] posterior=False, overshooting=True, lr4e-4
CUDA_VISIBLE_DEVICES=2 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 8 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 5 --lnr.optim.decay_steps 25000 --lnr.model.encoder_type slim --lnr.model.decoder_type slim --lnr.model.posterior_loss=False --lnr.model.overshooting_loss=True --lnr.optim.learning_rate 0.0004 --lnch.expname 11_1_21_t8_ph5_b32_lr4e-4_dr5e-1_st5e-1_ds25e3_etslim_dtslim_plFalse_osTrue &

[gauss1] posterior=True, overshooting=True, lr4e-4
CUDA_VISIBLE_DEVICES=3 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot ball_data/whiteballpush/U-Dk4s0n2000t10_ab --lnch.model_type factorized_world_model --lnr.sess.num_frames 8 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 5 --lnr.optim.decay_steps 25000 --lnr.model.encoder_type slim --lnr.model.decoder_type slim --lnr.model.posterior_loss=True --lnr.model.overshooting_loss=True --lnr.optim.learning_rate 0.0004 --lnch.expname 11_1_21_t8_ph5_b32_lr4e-4_dr5e-1_st5e-1_ds25e3_etslim_dtslim_plTrue_osTrue &

11:13am
[geb] finger
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot dmc_data/dw_fwm/dw_finger_easy_fwm_b32_t3_ph1_st5e-2 --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.decay_steps 25000 --lnch.expname 11_1_21_finger_t3_ph7_b32_lr1e-4_dr5e-1_st5e-1_ds25e3_etslim_dtslim_plFalse_osTrue &

[gauss1] finger: temp 0.1
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot dmc_data/debug/t_dmc_finger_turn_easy --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.decay_steps 25000 --lnr.model.temp 0.1 --lnch.expname 11_1_21_finger_t3_ph7_b32_lr1e-4_dr5e-1_st1e-1_ds25e3_etslim_dtslim_plFalse_osTrue &

[gauss1] finger: temp 0.05
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python train_slot_attention.py --lnch.dataroot dmc_data/debug/t_dmc_finger_turn_easy --lnch.model_type factorized_world_model --lnr.sess.num_frames 3 --lnr.optim.batch_size 32 --lnr.sess.pred_horizon 7 --lnr.optim.decay_steps 25000 --lnr.model.temp 0.05 --lnch.expname 11_1_21_finger_t3_ph7_b32_lr1e-4_dr5e-1_st5e-2_ds25e3_etslim_dtslim_plFalse_osTrue &

"""

