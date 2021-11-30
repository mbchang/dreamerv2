"""
reward head
actor head
critic head
"""
import einops as eo
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as tkl

from utils import *
import transformer


class SlotRewardHead(tkl.Layer):
    @staticmethod
    def defaults_debug():
        debug_args = SlotRewardHead.defaults()
        debug_args.head = transformer.TransformerDecoder.rew_defaults_debug()
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            head=transformer.TransformerDecoder.rew_defaults()
            ))
        return default_args

    def __init__(self, slot_size, cfg):
        super().__init__()
        self.head = transformer.TransformerDecoder(slot_size, cfg.head)
        self.out = linear(slot_size, 1)


    def call(self, slots):
        """
        x: B, T, K, D
        """
        x = bottle(self.head)(slots, slots)  # later we will just replace this with cross attention
        x = eo.reduce(x, 'b t k d -> b t d', 'mean')
        rew = self.out(x)
        return rew