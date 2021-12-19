import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent))

from utils import *

from einops import rearrange
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as tkl

class MultiHeadAttention(tkl.Layer):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn_dropout = tkl.Dropout(dropout)
        self.output_dropout = tkl.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    
    def call(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """        
        q = rearrange(self.proj_q(q), 'b l (head k) -> b head l k', head=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (head k) -> b head t k', head=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (head v) -> b head t v', head=self.num_heads)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = tf.einsum('bhlk,bhtk->bhlt', q, k)
        
        if attn_mask is not None:
            attn += (attn_mask * -1e9)  
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn)
        
        output = tf.einsum('hblt,hbtv->hblv', attn, v)
        output = rearrange(output, 'b head l v -> b l (head v)')
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output


class PositionalEncoding(tkl.Layer):

    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = tkl.Dropout(dropout)
        self.pe = self.add_weight(
          initializer="truncated_normal",
          shape=[1, max_len, d_model])

    def call(self, input, training=False):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, :T], training=training)


class TransformerEncoderBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1., is_first=False):
        super().__init__()
        
        self.is_first = is_first
        
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))
    
    
    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        raise NotImplementedError
        if self.is_first:
            input = self.attn_layer_norm(input)
            x = self.attn(input, input, input)
            input = input + x
        else:
            x = self.attn_layer_norm(input)
            x = self.attn(x, x, x)
            input = input + x
        
        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerEncoder(nn.Module):
    
    def __init__(self, num_blocks, d_model, num_heads, dropout=0.):
        super().__init__()
        
        if num_blocks > 0:
            gain = (2 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=True)] +
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=False)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    
    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        raise NotImplementedError
        for block in self.blocks:
            input = block(input)
        
        return self.layer_norm(input)


class TransformerDecoderBlock(tkl.Layer):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1., masked=False, is_first=False):
        super().__init__()
        
        self.is_first = is_first
        self.masked = masked
        
        self.self_attn_layer_norm = tkl.LayerNormalization(epsilon=1e-5)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.encoder_decoder_attn_layer_norm = tkl.LayerNormalization(epsilon=1e-5)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.ffn_layer_norm = tkl.LayerNormalization(epsilon=1e-5)
        self.ffn = tf.keras.Sequential([
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            tkl.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            tkl.Dropout(dropout)
            ])
    
    
    def call(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        if self.masked:
            T = input.shape[1]
            mask = tf.experimental.numpy.triu(tf.ones((T, T)), k=1)  # float, not bool
        else:
            mask = None
        
        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input, mask)
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, mask)
            input = input + x
        
        x = self.encoder_decoder_attn_layer_norm(input)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        input = input + x
        
        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerDecoder(tkl.Layer):

    @staticmethod
    def obs_defaults_debug():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=2,
            num_heads=2,
            dropout=0.1,
            masked=True,
            ))
        return default_args

    @staticmethod
    def obs_defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=4,
            num_heads=4,
            dropout=0.1,
            masked=True,
            ))
        return default_args

    @staticmethod
    def dyn_defaults_debug():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=2,
            num_heads=2,
            dropout=0.1,
            masked=False,
            ))
        return default_args

    @staticmethod
    def dyn_defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=4,
            num_heads=4,
            dropout=0.1,
            masked=False,
            ))
        return default_args

    @staticmethod
    def head_defaults_debug():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=1,
            num_heads=1,
            dropout=0.1,
            masked=False,
            ))
        return default_args

    @staticmethod
    def small_head_defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=2,
            num_heads=2,
            dropout=0.1,
            masked=False,
            ))
        return default_args

    @staticmethod
    def med_head_defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=4,
            num_heads=4,
            dropout=0.1,
            masked=False,
            ))
        return default_args

    @staticmethod
    def one_block_one_head_defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=1,
            num_heads=1,
            dropout=0.1,
            masked=False,
            ))
        return default_args

    @staticmethod
    def two_blocks_eight_heads_defaults():
        default_args = ml_collections.ConfigDict(dict(
            num_blocks=2,
            num_heads=8,
            dropout=0.1,
            masked=False,
            ))
        return default_args
    
    def __init__(self, d_model, cfg):
        super().__init__()

        num_blocks = cfg.num_blocks
        num_heads = cfg.num_heads
        dropout = cfg.dropout
        masked = cfg.masked

        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = [TransformerDecoderBlock(d_model, num_heads, dropout, gain, masked, is_first=True)] + [TransformerDecoderBlock(d_model, num_heads, dropout, gain, masked, is_first=False) for _ in range(num_blocks - 1)]
        else:
            self.blocks = []
        
        self.layer_norm = tkl.LayerNormalization(epsilon=1e-5)
    
    
    def call(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            input = block(input, encoder_output)
        
        return self.layer_norm(input)

