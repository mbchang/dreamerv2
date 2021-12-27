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
    def savi_defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_iterations=2,
            num_slot_heads=1,
            epsilon=1e-8,
            temp=0.5  # or we can make this temp=1.0
            ))
        return default_args
    
    def __init__(self, slot_size, cfg):
        super().__init__()
        
        self.num_iterations = cfg.num_iterations
        self.slot_size = slot_size
        self.epsilon = cfg.epsilon
        self.num_heads = cfg.num_slot_heads
        self.temp = cfg.temp

        self.norm_inputs = tkl.LayerNormalization(epsilon=1e-5)
        self.norm_slots = tkl.LayerNormalization(epsilon=1e-5)
        self.norm_mlp = tkl.LayerNormalization(epsilon=1e-5)
        
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

    def call(self, inputs, slots):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].

        inputs = self.norm_inputs(inputs)
        k = eo.rearrange(self.project_k(inputs), 'b t (head d) -> b head t d', head=self.num_heads)
        v = eo.rearrange(self.project_v(inputs), 'b t (head d) -> b head t d', head=self.num_heads)

        k = ((self.slot_size // self.num_heads) ** (-0.5) / self.temp) * k
        
        # Multiple rounds of attention.
        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = eo.rearrange(self.project_q(slots), 'b s (head d) -> b head s d', head=self.num_heads)
            attn_logits = tf.einsum('bhtd,bhsd->bhts', k, q)

            # import ipdb; ipdb.set_trace(context=20)
            # dtype = prec.global_policy().compute_dtype

            dtype = attn_logits.dtype
            attn_logits = tf.cast(attn_logits, tf.float32)


            attn = eo.rearrange(
                 tf.nn.softmax(eo.rearrange(attn_logits, 'b h t s -> b t (h s)'), axis=-1),
                 'b t (h s) -> b h t s', h=self.num_heads
                )

            # attn = eo.rearrange(
            #     tkl.Softmax(dtype='float32', axis=-1)(eo.rearrange(attn_logits, 'b h t s -> b t (h s)')),
            #     'b t (h s) -> b h t s', h=self.num_heads
            # )

            attn_vis = eo.reduce(attn, 'b h t s -> b t s', 'sum')
            
            # Weighted mean.
            attn = attn + self.epsilon
            """
            nan happened here at attn[-4] (the -4th example in the batch)

            <tf.Tensor: shape=(1, 256, 5), dtype=float16, numpy=
            array([[[0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    ...,
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.]]], dtype=float16)>

            then, for tf.math.reduce_sum(attn, axis=-2, keepdims=True), we had (we assigned the variable bb to that)

            ipdb> bb[:,0,0]
            <tf.Tensor: shape=(16, 5), dtype=float16, numpy=
            array([[3.9673e-04, 3.5840e+00, 5.1719e+01, 1.9725e+02, 1.4316e+00],
                   [2.5588e+02, 1.9073e-02, 4.7485e-02, 2.2675e-02, 9.3460e-03],
                   [7.6123e-01, 4.4116e-01, 1.0891e-03, 2.2275e+02, 2.9422e+01],
                   [8.8196e-02, 2.5412e+02, 1.0566e+00, 1.0699e-01, 6.3672e-01],
                   [5.1308e-04, 8.0764e-05, 2.0728e-01, 1.3804e-04, 2.5600e+02],
                   [1.1387e-03, 4.3082e-04, 4.1188e+01, 5.1956e-03, 2.1862e+02],
                   [2.5475e+02, 7.4196e-03, 3.5703e+00, 3.9978e-02, 4.5800e-04],
                   [3.4928e-05, 3.2656e+00, 1.0156e-01, 2.5338e+02, 2.8076e-02],
                   [9.2834e-02, 3.6438e-02, 2.3560e-02, 2.5588e+02, 4.8518e-05],
                   [4.1504e-02, 2.5588e+02, 6.2622e-02, 1.4183e-02, 1.2964e-01],
                   [2.0398e-01, 1.4270e-01, 2.4707e-01, 2.5488e+02, 5.4541e-01],
                   [3.3569e-04, 5.3673e-03, 2.8229e-04, 2.5512e+02, 1.8457e+00],
                   [2.9316e-03, 2.5600e+02, 1.5795e-05, 0.0000e+00, 2.3842e-07],
                   [1.1873e-04, 2.6709e-01, 2.5575e+02, 4.5121e-05, 4.4159e-02],
                   [2.5366e-01, 7.1094e+00, 2.5212e+02, 1.1814e-04, 4.2081e-05],
                   [1.8740e-04, 2.4155e-02, 1.7500e-04, 4.6344e+01, 2.1262e+02]],
                  dtype=float16)>

            so the denominator was 0. Why was the denominaotr 0?

            Perhaps the epsilon was not big enough. Because it was just summing over a bunch of zeros.

            I've found the problem.
            """

            attn = attn / tf.math.reduce_sum(attn, axis=-2, keepdims=True)  # nan appeared here. We do not want to divide by 0. THis means that attn should not round to 0. This means that attn should be float32. Can we convert back to float16 after this step?

            attn = tf.cast(attn, dtype)



            updates = tf.einsum('bhts,bhtd->bhsd', attn, v)
            updates = eo.rearrange(updates, 'b h s d -> b s (h d)')

            # Slot update.
            slots, _ = bottle(self.gru)(updates, slots_prev)
            slots = slots + self.mlp(self.norm_mlp(slots))

            try:
              tf.debugging.check_numerics(slots, 'slots')
            except Exception as e:
              lgr.debug(e)
              import ipdb; ipdb.set_trace(context=20)
        
        return slots, attn_vis

class SlotAttentionWithReset(SlotAttention):
    @staticmethod
    def defaults_debug():
        debug_args = SlotAttentionWithReset.defaults()
        debug_args.num_iterations = 2
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_iterations=3,
            num_slots=3,
            num_slot_heads=1,
            epsilon=1e-8,
            temp=1.0
            ))
        return default_args

    @staticmethod
    def dyn_defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_iterations=3,
            num_slots=3,
            num_slot_heads=1,
            epsilon=1e-8,
            temp=0.5
            ))
        return default_args

    def __init__(self, slot_size, cfg):
        SlotAttention.__init__(self, slot_size, cfg)
        self.num_slots = cfg.num_slots

        # Parameters for Gaussian init (shared by all slots).
        self.slots_mu = self.add_weight(initializer="glorot_uniform", shape=[1, 1, self.slot_size])
        self.slots_log_sigma = self.add_weight(initializer="glorot_uniform", shape=[1, 1, self.slot_size])

    def reset(self, batch_size):
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal([batch_size, self.num_slots, self.slot_size])
        return slots