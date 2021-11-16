from utils import *
# from dvae import dVAE
import dvae
# from slot_attn import SlotAttention
import slot_attn
# from transformer import PositionalEncoding, TransformerDecoder
import transformer
import utils

from einops import rearrange, repeat
from loguru import logger as lgr
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as layers
import time
import wandb

class SlotModel(layers.Layer):

    def __init__(self, args):
        super().__init__()

        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.num_tokens = (args.image_size // 4) ** 2

        # obs encoder
        self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
        self.positional_encoder = transformer.PositionalEncoding(1 + self.num_tokens, args.d_model, args.dropout)
        self.layer_norm = tkl.LayerNormalization(epsilon=1e-5)
        self.token_mlp = tf.keras.Sequential([
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            tkl.ReLU(),
            linear(args.d_model, args.d_model)])

        # recurrent: replace this with rssm
        self.slot_attn = slot_attn.SlotAttention(args.slot_size, args.slot_attn)
        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)

        # decoder
        self.tf_dec = transformer.TransformerDecoder(self.num_tokens, args.d_model, args.obs_transformer)
        self.out = linear(args.d_model, args.vocab_size, bias=False)

        self.training = False

    def embed_tokens(self, tokens):
        emb_input = self.dictionary(tokens)
        emb_input = self.positional_encoder(emb_input, training=self.training)
        emb_input = self.token_mlp(self.layer_norm(emb_input))
        return emb_input

    def apply_slot_attn(self, emb_input):
        slots = self.slot_attn.reset(emb_input.shape[0])
        slots, attns = self.slot_attn(emb_input[:, 1:], slots)
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

    @staticmethod
    @tf.function
    def loss_and_grad(slot_model, z_transformer_input, z_transformer_target):
        with tf.GradientTape() as tape:
            attns, cross_entropy = slot_model(z_transformer_input, z_transformer_target)
        gradients = tape.gradient(cross_entropy, slot_model.trainable_weights)
        return attns, cross_entropy, gradients

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
    attns = repeat(attns, 'b k h w -> b k c h w', c=3)  # 3 is img_channels
    attns = rearrange(image, 'b c h w -> b 1 c h w') * attns + 1. - attns
    return attns


class SLATE(layers.Layer):

    # optim
    # training
    # model args

    @staticmethod
    def get_default_args():
        default_args = ml_collections.ConfigDict(dict(
            log_interval=800,

            image_size=64,
            batch_size=50,

            lr_dvae=3e-4,
            lr_main=1e-4,
            lr_warmup_steps=30000,
            lr_decay_factor=1.0,

            vocab_size=1024,
            d_model=192,
            dropout=0.1,
            obs_transformer=transformer.TransformerDecoder.get_obs_model_args(),

            slot_attn=slot_attn.SlotAttention.get_default_args(),
            slot_size=192,
            img_channels=3,
            pos_channels=4,

            tau_start=1.0,
            tau_final=0.1,
            tau_steps=30000,

            hard=False,
            ))
        return default_args

    def __init__(self, args):
        super().__init__()

        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        self.dvae = dvae.dVAE(args.vocab_size, args.img_channels)
        self.dvae_optimizer = tf.keras.optimizers.Adam(args.lr_dvae, epsilon=1e-08)

        self.slot_model = SlotModel(args)
        self.main_optimizer = tf.keras.optimizers.Adam(args.lr_main, epsilon=1e-08)

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
            recon,
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
            return recon_transformer, attns

        return recon_transformer

    # later replace this with train_args?
    def train_step(self, image, global_step, args):
        t0 = time.time()

        tau = utils.cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps)

        lr_warmup_factor = utils.linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        self.dvae_optimizer.lr = utils.f32(args.lr_decay_factor * args.lr_dvae)
        self.main_optimizer.lr = utils.f32(args.lr_decay_factor * lr_warmup_factor * args.lr_main)

        recon, z_hard, mse, gradients = dvae.dVAE.loss_and_grad(self.dvae, image, tf.constant(tau), args.hard)
        self.dvae_optimizer.apply_gradients(zip(gradients, self.dvae.trainable_weights))

        z_transformer_input, z_transformer_target = create_tokens(tf.stop_gradient(z_hard))

        attns, cross_entropy, gradients = SlotModel.loss_and_grad(self.slot_model, z_transformer_input, z_transformer_target)
        # NOTE: if we put this inside tf.function then the performance becomes very bad
        self.main_optimizer.apply_gradients(zip(gradients, self.slot_model.trainable_weights))

        loss = mse + cross_entropy

        _, _, H_enc, W_enc = z_hard.shape
        attns = overlay_attention(attns, image, H_enc, W_enc)

        if global_step % args.log_interval == 0:
            lgr.info('Train Step: {:3} \t Loss: {:F} \t MSE: {:F} \t Time: {:F}'.format(
                  global_step, loss.numpy(), mse.numpy(), time.time()-t0))

            wandb.log({
                'train/loss': loss.numpy(),
                'train/cross_entropy': cross_entropy.numpy(),
                'train/mse': mse.numpy(),
                'train/tau': tau,
                'train/lr_dvae': self.dvae_optimizer.lr.numpy(),
                'train/lr_main': self.main_optimizer.lr.numpy(),
                'train/itr': global_step
                }, step=global_step)

        return recon, attns, tau

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