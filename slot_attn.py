from utils import *

import einops as eo
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as tkl

class Factorized(tf.Module):
  def register_num_slots(self, num_slots):
    self.num_slots = num_slots
    for sm in self.submodules:
      if isinstance(sm, Factorized):
        sm.register_num_slots(num_slots)


class SlotAttention(tkl.Layer, Factorized):

    @staticmethod
    def get_default_args():
        default_args = ml_collections.ConfigDict(dict(
            num_iterations=3,
            num_slots=3,
            num_slot_heads=1,
            epsilon=1e-8
            ))
        return default_args
    
    def __init__(self, slot_size, cfg):
        super().__init__()
        
        self.num_iterations = cfg.num_iterations
        self.num_slots = cfg.num_slots
        # self.input_size = input_size
        self.slot_size = slot_size
        self.epsilon = cfg.epsilon
        self.num_heads = cfg.num_slot_heads

        self.norm_inputs = tkl.LayerNormalization(epsilon=1e-5)
        self.norm_slots = tkl.LayerNormalization(epsilon=1e-5)
        self.norm_mlp = tkl.LayerNormalization(epsilon=1e-5)

        # Parameters for Gaussian init (shared by all slots).
        self.slots_mu = self.add_weight(initializer="glorot_uniform", shape=[1, 1, self.slot_size])
        self.slots_log_sigma = self.add_weight(initializer="glorot_uniform", shape=[1, 1, self.slot_size])
        
        # Linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(None, slot_size, bias=False)
        self.project_v = linear(None, slot_size, bias=False)
        
        # Slot update functions.
        self.gru = tkl.GRUCell(slot_size)
        self.mlp = tf.keras.Sequential([
            linear(slot_size, slot_size, weight_init='kaiming'),
            tkl.ReLU(),
            linear(slot_size, slot_size)])

    def reset(self, batch_size):
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal([batch_size, self.num_slots, self.slot_size])
        return slots

    def call(self, inputs, slots):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].

        inputs = self.norm_inputs(inputs)
        k = eo.rearrange(self.project_k(inputs), 'b t (head d) -> b head t d', head=self.num_heads)
        v = eo.rearrange(self.project_v(inputs), 'b t (head d) -> b head t d', head=self.num_heads)

        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k
        
        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = eo.rearrange(self.project_q(slots), 'b s (head d) -> b head s d', head=self.num_heads)
            attn_logits = tf.einsum('bhtd,bhsd->bhts', k, q)
            attn = eo.rearrange(
                 tf.nn.softmax(eo.rearrange(attn_logits, 'b h t s -> b t (h s)'), axis=-1),
                 'b t (h s) -> b h t s', h=self.num_heads
                )
            attn_vis = eo.reduce(attn, 'b h t s -> b t s', 'sum')
            
            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / tf.math.reduce_sum(attn, axis=-2, keepdims=True)
            updates = tf.einsum('bhts,bhtd->bhsd', attn, v)
            updates = eo.rearrange(updates, 'b h s d -> b s (h d)')

            # Slot update.
            slots, _ = bottle(self.gru)(updates, slots_prev)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn_vis

# class SlotAttentionEncoder(tkl.Layer):
    
#     def __init__(self, slot_size, cfg):
#         super().__init__()
        
#         self.cfg = cfg
#         # self.input_channels = input_channels
#         self.slot_size = slot_size
        
#         # # Parameters for Gaussian init (shared by all slots).
#         # self.slots_mu = self.add_weight(initializer="glorot_uniform", shape=[1, 1, self.slot_size])
#         # self.slots_log_sigma = self.add_weight(initializer="glorot_uniform", shape=[1, 1, self.slot_size])
        
#         self.slot_attention = SlotAttention(slot_size, cfg)
    
    
#     def call(self, x):
#         # `image` has shape: [batch_size, img_channels, img_height, img_width].
#         # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
#         B, *_ = x.shape
#         # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].

#         # Slot Attention module.
#         # slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal([B, self.cfg.num_slots, self.slot_size])
#         slots = self.slot_attention.reset(B)
#         slots, attn = self.slot_attention(x, slots)
#         # `slots` has shape: [batch_size, num_slots, slot_size].
#         # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].

#         return slots, attn

