from utils import *
from dvae import dVAE
from slot_attn import SlotAttentionEncoder
from transformer import PositionalEncoding, TransformerDecoder

from einops import rearrange
import tensorflow as tf
import tensorflow.keras.layers as layers

class SlotModel(layers.Layer):
    def __init__(self, args):
        super().__init__()

        self.num_slots = args.num_slots
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        self.positional_encoder = PositionalEncoding(1 + (args.image_size // 4) ** 2, args.d_model, args.dropout)

        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.pos_channels,
            args.num_slot_heads)

        self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)

        self.tf_dec = TransformerDecoder(
            args.num_dec_blocks, (args.image_size // 4) ** 2, args.d_model, args.num_heads, args.dropout)

        self.out = linear(args.d_model, args.vocab_size, bias=False)

        self.training = False

    def embed_tokens(self, tokens):
        emb_input = self.dictionary(tokens)
        emb_input = self.positional_encoder(emb_input, training=self.training)
        return emb_input

    def apply_slot_attn(self, emb_input):
        slots, attns = self.slot_attn(emb_input[:, 1:])
        slots = self.slot_proj(slots)
        return slots, attns

    def parallel_decode(self, emb_input, slots):
        decoder_output = self.tf_dec(emb_input[:, :-1], slots)
        pred = self.out(decoder_output)
        return pred

    def autoregressive_decode(self, z_hard, slots, gen_len):
        B = slots.shape[0]

        # generate image tokens auto-regressively
        z_gen = tf.zeros(0, dtype=z_hard.dtype)
        z_transformer_input = tf.concat([
            tf.ones((B, 1, 1), dtype=z_hard.dtype),
            tf.zeros((B, 1, self.vocab_size), dtype=z_hard.dtype)
            ], axis=-1)
        for t in range(gen_len):
            decoder_output = self.tf_dec(
                self.positional_encoder(self.dictionary(z_transformer_input)),
                slots
            )
            z_next = tf.one_hot(tf.math.argmax(self.out(decoder_output)[:, -1:], axis=-1), depth=self.vocab_size)
            z_gen = z_next if t == 0 else tf.concat((z_gen, z_next), axis=1)
            z_transformer_input = tf.concat([
                z_transformer_input,
                tf.concat([tf.zeros_like(z_next[:, :, :1]), z_next], axis=-1)
            ], axis=1)

        return z_gen

    @tf.function
    def call(self, z_transformer_input, z_transformer_target):
        B = z_transformer_input.shape[0]
        emb_input = self.embed_tokens(z_transformer_input)
        slots, attns = self.apply_slot_attn(emb_input)
        pred = self.parallel_decode(emb_input, slots)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(tf.reshape(z_transformer_target * tf.nn.log_softmax(pred, axis=-1), (B, -1)), axis=-1))
        return attns, cross_entropy


    def train(self):
        self.training = True

    def eval(self):
        self.training = False



def create_tokens(z_hard):
    # target tokens for transformer
    z_transformer_target = rearrange(z_hard, 'b c h w -> b (h w) c')

    # add BOS token
    B, zhw, zc = z_transformer_target.shape
    z_transformer_input = tf.concat([tf.zeros((B, zhw, 1)), z_transformer_target], axis=-1)
    z_transformer_input = tf.concat([
        tf.concat([tf.ones((B, 1, 1)), tf.zeros((B, 1, zc))], axis=-1),
         z_transformer_input], axis=-2)
    return z_transformer_input, z_transformer_target


def overlay_attention(attns, image, H_enc, W_enc):
    B, C, H, W = image.shape
    attns = rearrange(attns, 'b hw k -> b k hw')
    attns = tf.repeat(tf.repeat(rearrange(attns, 'b k (h w) -> b k h w', h = H_enc, w=W_enc), H // H_enc, axis=-2), W // W_enc, axis=-1)
    attns = rearrange(image, 'b c h w -> b 1 c h w') * attns + 1. - attns
    return attns


class SLATE(layers.Layer):
    def __init__(self, args):
        super().__init__()

        self.num_slots = args.num_slots
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        self.dvae = dVAE(args.vocab_size, args.img_channels)
        self.slot_model = SlotModel(args)

        self.training = False

    def call(self, image: tf.Tensor, tau: tf.Tensor, hard: bool):
        """
        image: batch_size x img_channels x H x W
        """

        B, C, H, W = image.shape
        recon, z_hard, mse = self.dvae(image, tau, hard)
        _, _, H_enc, W_enc = z_hard.shape

        z_transformer_input, z_transformer_target = create_tokens(tf.stop_gradient(z_hard))
        attns, cross_entropy = self.slot_model(z_transformer_input, z_transformer_target)

        attns = overlay_attention(attns, image, H_enc, W_enc)

        return (
            tf.clip_by_value(recon, 0., 1.),
            cross_entropy,
            mse,
            attns
        )

    @tf.function
    def reconstruct_autoregressive(self, image: tf.Tensor, eval: bool=False):
        """
        image: batch_size x img_channels x H x W
        """

        gen_len = (image.shape[-1] // 4) ** 2

        B, C, H, W = image.shape

        z_logits = self.dvae.get_logits(image)
        _, _, H_enc, W_enc = z_logits.shape

        z_hard = self.dvae.mode(z_logits)

        one_hot_tokens, _ = create_tokens(z_hard)
        emb_input = self.slot_model.embed_tokens(one_hot_tokens)
        slots, attns = self.slot_model.apply_slot_attn(emb_input)
        z_gen = self.slot_model.autoregressive_decode(z_hard, slots, gen_len)

        z_gen = tf.cast(rearrange(z_gen, 'b (h w) d -> b d h w', h=H_enc, w=W_enc), tf.float32)

        recon_transformer = self.dvae.decoder(z_gen)

        attns = overlay_attention(attns, image, H_enc, W_enc)

        if eval:
            return tf.clip_by_value(recon_transformer, 0., 1.), attns

        return tf.clip_by_value(recon_transformer, 0., 1.)

    def train(self):
        self.training = True
        self.slot_model.train()

    def eval(self):
        self.training = False
        self.slot_model.eval()


class OneHotDictionary(layers.Layer):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = layers.Embedding(vocab_size, emb_size)

    def call(self, x):
        """
        x: B, N, vocab_size
        """
        tokens = tf.math.argmax(x, axis=-1)
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs