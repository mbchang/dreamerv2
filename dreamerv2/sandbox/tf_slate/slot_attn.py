from utils import *

import einops as eo
import tensorflow as tf
import tensorflow.keras.layers as tkl

class SlotAttention(tkl.Layer):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size, heads,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.num_heads = heads

        self.norm_inputs = tkl.LayerNormalization(epsilon=1e-5)
        self.norm_slots = tkl.LayerNormalization(epsilon=1e-5)
        self.norm_mlp = tkl.LayerNormalization(epsilon=1e-5)
        
        # Linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # Slot update functions.
        # self.gru = gru_cell(slot_size, slot_size)
        self.gru = tkl.GRUCell(slot_size)
        self.mlp = tf.keras.Sequential([
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            # nn.ReLU(),
            tkl.ReLU(),
            linear(mlp_hidden_size, slot_size)])

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
            
            # # Attention.
            q = eo.rearrange(self.project_q(slots), 'b s (head d) -> b head s d', head=self.num_heads)
            # attn_logits = torch.matmul(k, q.transpose(-1, -2))                             # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn_logits = tf.einsum('bhtd,bhsd->bhts', k, q)
            attn = eo.rearrange(
                 tf.nn.softmax(eo.rearrange(attn_logits, 'b h t s -> b t (h s)'), axis=-1),
                 'b t (h s) -> b h t s', h=self.num_heads
                )
            attn_vis = eo.reduce(attn, 'b h t s -> b t s', 'sum')
            
            # # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / tf.math.reduce_sum(attn, axis=-2, keepdims=True)
            # updates = torch.matmul(attn.transpose(-1, -2), v)                              # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = tf.einsum('bhts,bhtd->bhsd', attn, v)
            updates = eo.rearrange(updates, 'b h s d -> b s (h d)')

            # Slot update.
            # slots = bottle(self.gru)(updates, slots_prev)
            slots, _ = bottle(self.gru)(updates, slots_prev)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn_vis

class SlotAttentionEncoder(tkl.Layer):
    
    def __init__(self, num_iterations, num_slots,
                 input_channels, slot_size, mlp_hidden_size, pos_channels, num_heads):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_channels = pos_channels

        # self.layer_norm = nn.LayerNorm(input_channels)
        self.layer_norm = tkl.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            linear(input_channels, input_channels, weight_init='kaiming'),
            # nn.ReLU(),
            tkl.ReLU(),
            linear(input_channels, input_channels)])
        
        # Parameters for Gaussian init (shared by all slots).
        # self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
        # self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
        # nn.init.xavier_uniform_(self.slot_mu)
        # nn.init.xavier_uniform_(self.slot_log_sigma)
        self.slots_mu = self.add_weight(initializer="glorot_uniform", shape=[1, 1, self.slot_size])
        self.slots_log_sigma = self.add_weight(initializer="glorot_uniform", shape=[1, 1, self.slot_size])
        
        self.slot_attention = SlotAttention(
            num_iterations, num_slots,
            input_channels, slot_size, mlp_hidden_size, num_heads)
    
    
    def call(self, x):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        # B, *_ = x.size()
        B, *_ = x.shape
        x = self.mlp(self.layer_norm(x))
        # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].

        # Slot Attention module.
        # slots = x.new_empty(B, self.num_slots, self.slot_size).normal_()
        # slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots
        slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal([B, self.num_slots, self.slot_size])
        slots, attn = self.slot_attention(x, slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].
        
        return slots, attn

