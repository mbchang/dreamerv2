from einops import rearrange
import ml_collections
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common
import expl

from sandbox import causal_agent
from sandbox import machine
from sandbox import normalize as nmlz
import sandbox.tf_slate.utils as slate_utils
from sandbox.tf_slate import slot_model as sm


class SlotActorCritic(causal_agent.ActorCritic):
  @staticmethod
  def defaults_debug():
    debug_args = SlotActorCritic.defaults()
    debug_args.actor = sm.SlotHead.defaults_debug()
    debug_args.critic = sm.SlotHead.defaults_debug()
    return debug_args

  @staticmethod
  def defaults():
    default_args = ml_collections.ConfigDict(dict(
      actor=sm.SlotHead.defaults(),
      critic=sm.SlotHead.defaults(),
      use_slot_heads=True,
      ))
    return default_args

  def __init__(self, config, act_space, tfstep):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})
    #################################################################
    if self.config.slot_behavior.use_slot_heads:
      self.actor = sm.DistSlotHead(
        slot_size=self.config.dslate.slot_model.slot_size, 
        shape=act_space.shape[0], 
        dist_cfg=dict(dist=self.config.actor.dist, min_std=self.config.actor.min_std),
        cfg=self.config.slot_behavior.actor)
      self.critic = sm.DistSlotHead(
        slot_size=self.config.dslate.slot_model.slot_size, 
        shape=[], 
        dist_cfg=dict(dist=self.config.critic.dist),
        cfg=self.config.slot_behavior.critic)
      if self.config.slow_target:
        self._target_critic = sm.DistSlotHead(
          slot_size=self.config.dslate.slot_model.slot_size, 
          shape=[], 
          dist_cfg=dict(dist=self.config.critic.dist),
          cfg=self.config.slot_behavior.critic)
        self._updates = tf.Variable(0, tf.int64)
      else:
        self._target_critic = self.critic
    #################################################################
    else:
      self.actor = common.MLP(act_space.shape[0], **self.config.actor)
      self.critic = common.MLP([], **self.config.critic)
      if self.config.slow_target:
        self._target_critic = common.MLP([], **self.config.critic)
        self._updates = tf.Variable(0, tf.int64)
      else:
        self._target_critic = self.critic
    #################################################################
    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)


"""
12/4/21

python dreamerv2/train.py --configs debug dslate --task dmc_manip_lift_large_box --agent causal --dataset.length 8 --dataset.batch 3 --eval_dataset.length 10 --logdir runs/debug_wandb --jit False --steps 125 --wm_only=False --delay_train_behavior_by 5
"""