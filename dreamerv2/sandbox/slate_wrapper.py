import datetime
import einops as eo
from loguru import logger as lgr
import ml_collections
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(__file__))
import tensorflow as tf
import wandb

import common

from sandbox import causal_agent
import sandbox.tf_slate.slate as slate
import sandbox.tf_slate.utils as utils
from sandbox import normalize as nmlz
import sandbox.slot_attention_utils as sa_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DynamicSlateWrapperForDreamer(causal_agent.WorldModel):
  """
  """
  def __init__(self, config, obs_space, tfstep):
    self.config = config
    self.defaults = ml_collections.ConfigDict(self.config.dslate)
    self.model = slate.DynamicSLATE(self.defaults)


  def train(self, data, state=None):
    """
      reward (B, T)
      is_first (B, T)
      is_last (B, T)
      is_terminal (B, T)
      image (B, T, H, W, C)
      orientations (B, T, D)
      height (B, T)
      velocity (B, T, V)
      action (B, T, A)

      # note that we may need to take this outside of tf.function though
    """
    data = self.preprocess(data)

    # TODO: make is_first flag the first action

    # do this for now

    # train step
    loss, outputs, mets = self.model.train_step(data)  
    # state is dummy
    state = None

    # metrics
    metrics = {
      'kl_loss': 0,
      'image_loss': mets['mse'],
      'reward_loss': 0,
      'discount_loss': 0,
      'model_kl': mets['cross_entropy'],
      'prior_ent': 0,
      'post_ent': 0,
      
      'slate/loss': loss,
      'slate/mse': mets['mse'],
      'slate/cross_entropy': mets['cross_entropy'],
      'slate/slot_model_lr': self.model.main_optimizer.lr,
      'slate/dvae_lr': self.model.dvae_optimizer.lr,
      'slate/itr': self.model.step,
      'slate/tau': outputs['iterates']['tau'],
    }

    # outputs is dummy
    outputs = None

    return state, outputs, metrics

  # @tf.function# --> if I turn this on I can't do video.numpy()
  def report(self, data):
    report = {}
    data = self.preprocess(data)

    name = 'image'
    seed_steps = self.config.eval_dataset.seed_steps

    iterates = self.model.get_iterates(self.model.step.numpy())

    outs, mets = self.model(data, tf.constant(iterates['tau']), True)


    image = data['image']
    B, T, *_ = image.shape
    image = eo.rearrange(image, 'b t h w c -> (b t) c h w')

    if not self.config.dslate.vis_rollout:
      rollout, _, _ = slate.SLATE.reconstruct_autoregressive(self.model, image)
    else:
      # data['image']: TensorShape([6, 10, 64, 64, 3])
      # outs['dvae']['recon']: TensorShape([60, 3, 64, 64])
      # outs['slot_model']['attns']: TensorShape([60, 256, 5])
      # rollout_output['video']: TensorShape([6, 10, 3, 64, 64])
      rollout_output, rollout_metrics = self.model.rollout(data, seed_steps, self.config.eval_dataset.length-seed_steps)
      rollout = eo.rearrange(rollout_output['video'], 'b t c h w -> (b t) c h w')

    vis_recon = slate.SLATE.visualize(
      image, 
      outs['slot_model']['attns'], 
      outs['dvae']['recon'], 
      rollout,
      lambda x: tf.clip_by_value(nmlz.uncenter(x), 0., 1.))  # c (b h) (n w)
    video = eo.rearrange(vis_recon, '(b t) n c h w -> t (b h) (n w) c', b=B)

    """
    ok what do we want to see?
    first column: ground truth
    second column: dvae reconstruction
    third column: 
      reconstruct: autoregressive decode
      imagine: autoregressive decode
    attentions:
      reconstruct: yes
      imagine: no for now

    primitives we need:
    - encode
    - autoregressive decode
    """

    #######################################





    report[f'openl_{name}'] = video
    # report[f'recon_loss_{name}'] = rollout_metrics['reconstruct']
    # report[f'imag_loss_{name}'] = rollout_metrics['imagine']

    logdir = (Path(self.config.logdir) / Path(self.config.expdir)).expanduser()
    save_path = os.path.join(logdir, f'{self.model.step.numpy()}')
    lgr.info(f'Save gif to {save_path}. Video hash: {utils.hash_sha1(video)}')
    # lgr.info(f'Save gif to {save_path}. Video hash: {utils.hash_50(video)}')
    sa_utils.save_gif(sa_utils.add_border(video.numpy(), seed_steps), save_path)
    np.save(f'vis_rollout-{self.config.dslate.vis_rollout}.npy', video.numpy())
    return report


  # # @tf.function# --> if I turn this on I can't do video.numpy()
  # def report(self, data):
  #   report = {}
  #   data = self.preprocess(data)

  #   name = 'image'
  #   seed_steps = self.config.eval_dataset.seed_steps

  #   rollout_output, rollout_metrics = self.model.rollout(batch=data, seed_steps=seed_steps, pred_horizon=self.config.eval_dataset.length-seed_steps)
  #   video = self.model.visualize(rollout_output) # (10, 512, 384, 3) = t h (b w) c

  #   report[f'openl_{name}'] = video
  #   report[f'recon_loss_{name}'] = rollout_metrics['reconstruct']
  #   report[f'imag_loss_{name}'] = rollout_metrics['imagine']

  #   logdir = (Path(self.config.logdir) / Path(self.config.expdir)).expanduser()
  #   save_path = os.path.join(logdir, f'{self.step.numpy()}')
  #   lgr.info(f'save gif to {save_path}')
  #   utils.save_gif(utils.add_border(video.numpy(), seed_steps), save_path)
  #   return report








"""
python dreamerv2/train.py --configs debug slate --task dmc_cheetah_run --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --fwm.monitoring.log_every 1 --fwm.optim.warmup_steps 0 --fwm.optim.decay_steps 1 --fwm.optim.min_lr 8e-5 --steps 125 --fwm.model.dim 32


11-19-21 11:43am After I added slate defaults and got rid of fwm defaults:

python dreamerv2/train.py --configs debug slate --task dmc_cheetah_run --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --steps 125

11-21-21: 9:28am: 
python dreamerv2/train.py --configs debug dslate --task dmc_manip_lift_large_box --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --steps 125
"""