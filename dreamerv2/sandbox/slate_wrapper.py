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
    """
    data = self.preprocess(data)

    image = eo.rearrange(data['image'], 'b t h w c -> (b t) c h w')
    loss, outputs, mets = self.model.train_step(image)
    # state is dummy
    state = None

    # metrics
    metrics = {
      'kl_loss': 0,
      'image_loss': mets['dvae']['mse'],
      'reward_loss': 0,
      'discount_loss': 0,
      'model_kl': mets['slot_model']['cross_entropy'],
      'prior_ent': 0,
      'post_ent': 0,
      
      'slate/loss': loss,
      'slate/mse': mets['dvae']['mse'],
      'slate/cross_entropy': mets['slot_model']['cross_entropy'],
      'slate/slot_model_lr': self.model.main_optimizer.lr,
      'slate/dvae_lr': self.model.dvae_optimizer.lr,
      'slate/itr': self.model.step,
      'slate/tau': outputs['iterates']['tau'],
    }

    # outputs is dummy
    outputs = None

    return state, outputs, metrics

  def report(self, data):
    report = {}
    data = self.preprocess(data)

    name = 'image'
    seed_steps = self.config.eval_dataset.seed_steps

    iterates = self.model.get_iterates(self.model.step.numpy())

    image = eo.rearrange(data['image'], 'b t h w c -> (b t) c h w')
    loss, outs, mets = self.model(image, tf.constant(iterates['tau']), True)

    gen_img, _, _ = self.model.reconstruct_autoregressive(image)

    vis_recon = slate.SLATE.visualize(
      image, 
      outs['slot_model']['attns'], 
      outs['dvae']['recon'], 
      gen_img,
      lambda x: tf.clip_by_value(nmlz.uncenter(x), 0., 1.))  # c (b h) (n w)
    video = eo.rearrange(vis_recon, '(b t) n c h w -> t (b h) (n w) c', b=data['action'].shape[0])

    report[f'openl_{name}'] = eo.rearrange(video, 't h w c -> 1 (t h) w c')
    # report[f'recon_loss_{name}'] = rollout_metrics['reconstruct']
    # report[f'imag_loss_{name}'] = rollout_metrics['imagine']

    logdir = (Path(self.config.logdir) / Path(self.config.expdir)).expanduser()
    save_path = os.path.join(logdir, f'{self.model.step.numpy()}')
    lgr.info(f'Save png to {save_path}. Video hash: {utils.hash_sha1(video)}')
    plt.imsave(f'{save_path}.png', video[0].numpy())  # first timestep #eo.rearrange())#, 't h w c -> (t h) w c'))
    return report


class DynamicSlateWrapperForDreamer(causal_agent.WorldModel):
  """
  """
  def __init__(self, config, obs_space, tfstep):
    self.config = config
    self.defaults = ml_collections.ConfigDict(self.config.dslate)
    self.model = slate.DynamicSLATE(self.config.dataset.length, self.defaults)

    # integrate for actor and critic
    self.rssm = self.model.slot_model
    self.heads = self.model.slot_model.heads

  def encoder(self, data):
    assert len(data['image'].shape) == 4  # this way we don't bottle
    permute = lambda x: eo.rearrange(x, '... h w c -> ... c h w')
    image = permute(data['image'])

    tau = utils.tf_cosine_anneal(
        step=tf.cast(self.model.step, tf.float32),
        start_value=tf.cast(self.model.args.dvae.tau_start, tf.float32),
        final_value=tf.cast(self.model.args.dvae.tau_final, tf.float32),
        start_step=tf.cast(0, tf.float32),
        final_step=tf.cast(self.model.args.dvae.tau_steps, tf.float32))

    z_hard = self.model.dvae.sample_encode(image, tau, True)
    z_input, _ = slate.create_tokens(tf.stop_gradient(z_hard))
    emb_input = self.model.slot_model.embed_tokens(z_input)
    return emb_input

  def log_weights(self, step):
    # import time
    # t0 = time.time()
    for v in self.model.dvae.variables:
      tf.summary.histogram(f'dvae/{v.name}', v.value(), step=step.value)
    for v in self.model.slot_model.variables:
      tf.summary.histogram(f'slot_model/{v.name}', v.value(), step=step.value)
    # lgr.info(f'Writing histogram took {time.time()-t0} seconds.')    

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

    # do this for now

    # train step
    loss, outputs, mets = self.model.train_step(data)  
    # state is dummy
    state = None

    # metrics
    metrics = {
      'kl_loss': 0,
      'image_loss': mets['dvae']['mse'],
      'reward_loss': mets['slot_model']['rew_loss'],
      'discount_loss': 0,
      'model_kl': mets['slot_model']['cross_entropy'],
      'prior_ent': 0,
      'post_ent': 0,
      
      'slate/loss': loss,
      'slate/mse': mets['dvae']['mse'],
      'slate/cross_entropy': mets['slot_model']['cross_entropy'],
      'slate/slot_model_lr': self.model.main_optimizer.lr,
      'slate/dvae_lr': self.model.dvae_optimizer.lr,
      'slate/itr': self.model.step,
      'slate/tau': outputs['iterates']['tau'],
      'slate/num_frames': outputs['iterates']['num_frames'],
      'slate/lr_decay_factor': outputs['iterates']['lr_decay_factor'],
      'slate/consistency': mets['slot_model']['consistency'],
    }

    # outputs is dummy
    outputs = {
      'post': {'deter': outputs['slot_model']['post']}
    }

    return state, outputs, metrics

  # @tf.function# --> if I turn this on I can't do video.numpy()
  def report(self, data):
    report = {}
    data = self.preprocess(data)

    name = 'image'
    seed_steps = self.config.eval_dataset.seed_steps

    iterates = self.model.get_iterates(self.model.step.numpy())

    loss, outs, mets = self.model(data, tf.constant(iterates['tau']), True)


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
    report[f'slate/recon_cross_entropy'] = rollout_metrics['recon']
    if 'imag' in rollout_metrics:
      report[f'slate/imag_cross_entropy'] = rollout_metrics['imag']

    logdir = (Path(self.config.logdir) / Path(self.config.expdir)).expanduser()
    save_path = os.path.join(logdir, f'{self.model.step.numpy()}')
    lgr.info(f'Save gif to {save_path}. Video hash: {utils.hash_sha1(video)}')
    # lgr.info(f'Save gif to {save_path}. Video hash: {utils.hash_50(video)}')
    sa_utils.save_gif(sa_utils.add_border(video.numpy(), seed_steps), save_path)
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

12-1-21:
python dreamerv2/train.py --configs debug dslate --task dmc_manip_lift_large_box --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit True --steps 125 --wm_only=False


12-7-21:

python dreamerv2/train.py --configs debug dslate --task dmc_manip_lift_large_box --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --steps 125 --wm_only=False --delay_train_behavior_by 5 --slot_behavior.use_slot_heads False

python dreamerv2/train.py --configs debug dslate --task dmc_manip_lift_large_box --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --steps 125 --wm_only=False --delay_train_behavior_by 5 --slot_behavior.use_slot_heads True
"""