from utils import *
import slot_attn
import transformer

import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as layers

class SlotModel(layers.Layer):

    @staticmethod
    def defaults_debug():
        debug_args = SlotModel.defaults()
        debug_args.d_model = 16
        debug_args.slot_size = 16
        debug_args.lr_warmup_steps = 3
        debug_args.obs_transformer = transformer.TransformerDecoder.obs_defaults_debug()
        debug_args.slot_attn=slot_attn.SlotAttention.defaults_debug()
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            lr=1e-4,
            lr_warmup_steps=30000,

            d_model=192,
            dropout=0.1,

            obs_transformer=transformer.TransformerDecoder.obs_defaults(),

            slot_attn=slot_attn.SlotAttention.defaults(),
            slot_size=192,
            ))
        return default_args

    def __init__(self, vocab_size, num_tokens, args):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = args.d_model
        self.num_tokens = num_tokens

        # obs encoder
        self.dictionary = OneHotDictionary(self.vocab_size + 1, args.d_model)
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
        self.out = linear(args.d_model, self.vocab_size, bias=False)

        self.training = False

    def embed_tokens(self, tokens):
        emb_input = self.dictionary(tokens)
        emb_input = self.positional_encoder(emb_input, training=self.training)
        return emb_input

    def apply_slot_attn(self, emb_input, slots=None):
        if slots is None:
            slots = self.slot_attn.reset(emb_input.shape[0])
        slots, attns = self.slot_attn(self.token_mlp(self.layer_norm(emb_input[:, 1:])), slots)
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
