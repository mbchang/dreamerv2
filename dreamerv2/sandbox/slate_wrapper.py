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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SlateWrapperForDreamer(causal_agent.WorldModel):
  """
  """
  def __init__(self, config, obs_space, tfstep):
    self.config = config
    self.defaults = ml_collections.ConfigDict(self.config.slate)
    self.model = slate.SLATE(self.defaults)


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
    image = eo.rearrange(data['image'], 'b t h w c -> (b t) c h w')

    # train step
    loss, outputs, mets = self.model.train_step(image)  
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

    image = eo.rearrange(data['image'], 'b t h w c -> (b t) c h w')
    tau = utils.cosine_anneal(
        step=self.model.step.numpy(),
        start_value=self.model.args.dvae.tau_start,
        final_value=self.model.args.dvae.tau_final,
        start_step=0,
        final_step=self.model.args.dvae.tau_steps)

    (recon, cross_entropy, mse, attns, z_hard) = self.model(image, tf.constant(tau), True)
    vis_recon = utils.report(image, attns, recon, z_hard, self.model, lambda x: tf.clip_by_value(nmlz.uncenter(x), 0., 1.), n=self.config.eval_dataset.batch, prefix='REPORT', verbose=False)  # c (b h) (n w)

    report[f'openl_{name}'] = eo.rearrange(vis_recon, 'c h w -> 1 h w c')
    # report[f'recon_loss_{name}'] = rollout_metrics['reconstruct']
    # report[f'imag_loss_{name}'] = rollout_metrics['imagine']

    logdir = (Path(self.config.logdir) / Path(self.config.expdir)).expanduser()
    save_path = os.path.join(logdir, f'{self.model.step.numpy()}')
    lgr.info(f'save png to {save_path}')
    # utils.save_gif(utils.add_border(video.numpy(), seed_steps), save_path)
    plt.imsave(f'{save_path}.png', eo.rearrange(vis_recon.numpy(), 'c h w -> h w c'))

    return report



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

    image = eo.rearrange(data['image'], 'b t h w c -> (b t) c h w')
    tau = utils.cosine_anneal(
        step=self.model.step.numpy(),
        start_value=self.model.args.dvae.tau_start,
        final_value=self.model.args.dvae.tau_final,
        start_step=0,
        final_step=self.model.args.dvae.tau_steps)

    (recon, cross_entropy, mse, attns, z_hard) = self.model(image, tf.constant(tau), True)
    vis_recon = utils.report(image, attns, recon, z_hard, self.model, lambda x: tf.clip_by_value(nmlz.uncenter(x), 0., 1.), n=self.config.eval_dataset.batch, prefix='REPORT', verbose=False)  # c (b h) (n w)

    report[f'openl_{name}'] = eo.rearrange(vis_recon, 'c h w -> 1 h w c')
    # report[f'recon_loss_{name}'] = rollout_metrics['reconstruct']
    # report[f'imag_loss_{name}'] = rollout_metrics['imagine']

    logdir = (Path(self.config.logdir) / Path(self.config.expdir)).expanduser()
    save_path = os.path.join(logdir, f'{self.model.step.numpy()}')
    lgr.info(f'save png to {save_path}')
    # utils.save_gif(utils.add_border(video.numpy(), seed_steps), save_path)
    plt.imsave(f'{save_path}.png', eo.rearrange(vis_recon.numpy(), 'c h w -> h w c'))

    return report




"""
python dreamerv2/train.py --configs debug slate --task dmc_cheetah_run --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --fwm.monitoring.log_every 1 --fwm.optim.warmup_steps 0 --fwm.optim.decay_steps 1 --fwm.optim.min_lr 8e-5 --steps 125 --fwm.model.dim 32


11-19-21 11:43am After I added slate defaults and got rid of fwm defaults:

python dreamerv2/train.py --configs debug slate --task dmc_cheetah_run --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --steps 125
"""