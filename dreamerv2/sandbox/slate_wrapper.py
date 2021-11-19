import datetime
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
from sandbox import slot_attention_learners
from sandbox import slot_attention_utils as utils


class SlateWrapperForDreamer(causal_agent.WorldModel):
  """
  """
  def __init__(self, config, obs_space, tfstep):
    self.config = config
    self.defaults = ml_collections.ConfigDict(self.config.fwm)
    self.model = slot_attention_learners.FactorizedWorldModel(self.defaults.model)
    self.model.register_num_slots(self.defaults.sess.num_slots)

    self.optimizer = tf.keras.optimizers.Adam(self.defaults.optim.learning_rate, epsilon=1e-08)

    self.step = tf.Variable(0, trainable=False, name="fwm_step", dtype=tf.int64)
    self.min_lr = tf.Variable(self.defaults.optim.min_lr, trainable=False)

  def adjust_lr(self, step):
    if self.step < self.defaults.optim.warmup_steps:
      learning_rate = tf.cast(self.defaults.optim.learning_rate, tf.float32)
    else:
      learning_rate = self.defaults.optim.learning_rate * (self.defaults.optim.decay_rate ** (
          tf.cast(self.step-self.defaults.optim.warmup_steps, tf.float32) / tf.cast(self.defaults.optim.decay_steps, tf.float32)))
    learning_rate = tf.math.maximum(learning_rate, self.min_lr)
    return learning_rate


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
    """
    data = self.preprocess(data)

    # TODO: make is_first flag the first action

    # adjust learning rate
    self.optimizer.lr = self.adjust_lr(self.step)

    # train step
    loss, outputs, mets = self.model.train_step(data, self.optimizer)

    self.step.assign_add(1)

    # state is dummy
    state = None

    # outputs is dummy
    outputs = None

    # metrics
    metrics = {
      'kl_loss': 0,
      'image_loss': mets['posterior'],
      'reward_loss': 0,
      'discount_loss': 0,
      'model_kl': mets['initial_latent'] + mets['subsequent_latent'],  # are we averaging over time for subsequent latent?
      'prior_ent': 0,
      'post_ent': 0,
      
      'fwm/loss': loss,
      'fwm/learning_rate': self.optimizer.lr,
      'fwm/itr': self.step,
    }
    return state, outputs, metrics

  # @tf.function# --> if I turn this on I can't do video.numpy()
  def report(self, data):
    report = {}
    data = self.preprocess(data)

    name = 'image'
    seed_steps = self.config.eval_dataset.seed_steps

    rollout_output, rollout_metrics = self.model.rollout(batch=data, seed_steps=seed_steps, pred_horizon=self.config.eval_dataset.length-seed_steps)
    video = self.model.visualize(rollout_output)

    report[f'openl_{name}'] = video
    report[f'recon_loss_{name}'] = rollout_metrics['reconstruct']
    report[f'imag_loss_{name}'] = rollout_metrics['imagine']

    logdir = (Path(self.config.logdir) / Path(self.config.expdir)).expanduser()
    save_path = os.path.join(logdir, f'{self.step.numpy()}')
    lgr.info(f'save gif to {save_path}')
    utils.save_gif(utils.add_border(video.numpy(), seed_steps), save_path)
    return report



"""
python dreamerv2/train.py --configs debug slate --task dmc_cheetah_run --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --fwm.monitoring.log_every 1 --fwm.optim.warmup_steps 0 --fwm.optim.decay_steps 1 --fwm.optim.min_lr 8e-5 --steps 125 --fwm.model.dim 32


11-19-21 11:43am After I added slate defaults and got rid of fwm defaults:

python dreamerv2/train.py --configs debug slate --task dmc_cheetah_run --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --steps 125
"""