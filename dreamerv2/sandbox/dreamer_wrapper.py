import datetime
from loguru import logger as lgr
import ml_collections
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(__file__))
import tensorflow as tf
import wandb

import common

from sandbox import causal_agent
from sandbox import slot_attention_learners




"""
_wandb:
  desc: null
  value:
    cli_version: 0.12.6
    framework: keras
    is_jupyter_run: false
    is_kaggle_kernel: false
    python_version: 3.9.7
    start_time: 1635439930
    t:
      1:
      - 2
      - 3
      3:
      - 14
      - 16
      4: 3.9.7
      5: 0.12.6
      8:
      - 5

dataroot:
  desc: null
  value: ball_data/whiteballpush/U-Dk4s0n2000t10_ab

model_type:
  desc: null
  value: factorized_world_model

system:
  cpu:
  headless:

monitoring:
  expname
  jobtype
  log_every
  save_every
  subroot
  vis_every

"""



class FactorizedWorldModelWrapperForDreamer(causal_agent.WorldModel):
  """
    Ok, so for some reason 
      runs/model/balls_sa/balls_sa_ns5_t3_fwm seems to be better than 
      runs/dw_fwm_b32_t3_ph1.

      The only difference is that in runs/model/balls_sa/balls_sa_ns5_t3_fwm there is no learning rate decay.

      from train_slot_attention, I learned that it is better to have a constant learning rate in the beginning and the decay it later: t3_ph7_b32_lr1e-4_dr5e-1_st5e-1_imagpost_ds25e3_wuconstant_geb seems to be the best
  """


  @staticmethod
  def get_default_args():
      default_args = ml_collections.ConfigDict(dict(
        optim=ml_collections.ConfigDict(dict(
          batch_size=32,
          decay_rate=0.5,
          decay_steps=50000,  # or maybe 50000
          learning_rate=1e-4,
          num_train_steps=500000,
          warmup_steps=0,  # should probably be 10000
          )),  
          # maybe we should warmup, and then decay every 25000? we want to get to the initial dip, but once we hit that we'd want to decay it more aggressively.
        eval=ml_collections.ConfigDict(dict(
          pred_horizon=7)),
        model=ml_collections.ConfigDict(dict(
          resolution=(64, 64),
          temp=0.5)),
        sess=ml_collections.ConfigDict(dict(
          num_frames=3,
          num_slots=5,
          seed=0)),
        monitoring=ml_collections.ConfigDict(dict(
          log_every=100,
          # log_every=1,
          save_every=1000,
          vis_every=1000))
        ))
      return default_args


  def __init__(self, config, obs_space, tfstep):
    self.config = config
    self.defaults = FactorizedWorldModelWrapperForDreamer.get_default_args()

    self.model = slot_attention_learners.FactorizedWorldModel(
      num_slots=self.defaults.sess.num_slots,  # should be removed at some point
      **self.defaults.model)
    self.optimizer = tf.keras.optimizers.Adam(self.defaults.optim.learning_rate, epsilon=1e-08)

    self.step = 0

    wandb.init(
        config=self.defaults.to_dict(),  # will need to change this
        project='slot attention',
        dir=self.config.logdir,
        group=f'dreamer_wrapper_{Path(self.config.logdir).parent.name}',
        job_type='train',
        id=f'dw_{Path(self.config.logdir).name}_{datetime.datetime.now():%Y%m%d%H%M%S}'
        )

  def adjust_lr(self, step):
    # if self.step < self.defaults.optim.warmup_steps:
    #   learning_rate = self.defaults.optim.learning_rate * float(self.step) / self.defaults.optim.warmup_steps
    # else:
    #   learning_rate = self.defaults.optim.learning_rate
    # learning_rate = learning_rate * (self.defaults.optim.decay_rate ** (float(self.step) / self.defaults.optim.decay_steps))


    learning_rate = self.defaults.optim.learning_rate
    self.optimizer.lr = learning_rate

    # if self.step < self.defaults.optim.warmup_steps:
    #   learning_rate = self.defaults.optim.learning_rate
    # else:
    #   learning_rate = self.defaults.optim.learning_rate * (self.defaults.optim.decay_rate ** (float(self.step-self.defaults.optim.warmup_steps) / self.defaults.optim.decay_steps))

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

    # delete the first action
    data['action'] = data['action'][:, 1:]

    # adjust learning rate
    lr = self.adjust_lr(self.step)

    # train step
    loss, outputs, mets = self.model.train_step(data, self.optimizer)

    self.step += 1

    if self.step % self.defaults.monitoring.log_every == 0:
      lgr.info(f'step: {self.step}\tloss: {loss}\tlr: {lr}')

      wandb.log({
          f'train/itr': self.step,
          f'train/loss': loss,
          f'train/learning_rate': lr,
          }, step=self.step)


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
    }
    return state, outputs, metrics

  @tf.function
  def report(self, data):
    report = {}
    data = self.preprocess(data)
    # delete the first action
    data['action'] = data['action'][:, 1:]

    name = 'image'
    seed_steps = self.config.video_pred.seed_steps
    pred_horizon = self.config.dataset.length - self.config.video_pred.seed_steps
    report[f'openl_{name}'] = self.model.visualize(data, seed_steps=seed_steps, pred_horizon=pred_horizon)  # for now
    return report



"""
python dreamerv2/train_model.py --logdir runs/debug/model/slot2 --configs debug --task dmc_cheetah_run --agent causal --log_every 5 --dataset.batch 10 --video_pred.seed_steps 2 --dataset.length 5 --rssm.dynamics slim_cross_attention --rssm.update slot_attention --decoder_type slimmerslot --encoder_type slimmerslot --rssm.num_slots 2 --wm fwm --precision 32


python dreamerv2/train_model.py --logdir runs/debug/merge/default --configs debug --task balls_whiteball_push --agent causal --wm_only=True --precision 32 --wm fwm

10/28/21
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0 python dreamerv2/train_model.py --logdir runs/dw_fwm_b32_t3_ph1/sanity --configs dmc_vision --task balls_whiteball_push --agent causal --prefill 20000 --wm_only=True --precision 32 --dataset.batch 32 --dataset.length 3 --video_pred.seed_steps 2 --wm fwm &

10/29/21

"""