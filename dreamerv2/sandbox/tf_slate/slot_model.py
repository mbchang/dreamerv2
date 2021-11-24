from utils import *
import slot_attn
import transformer

import einops as eo
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as layers

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

        self.args = args
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
        self.tf_dec = transformer.TransformerDecoder(args.d_model, args.obs_transformer)
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

    def autoregressive_decode(self, slots):
        B = slots.shape[0]

        # generate image tokens auto-regressively
        z_gen = tf.zeros(0, dtype=tf.float32)
        z_input = tf.concat([
            tf.ones((B, 1, 1), dtype=tf.float32),
            tf.zeros((B, 1, self.vocab_size), dtype=tf.float32)
            ], axis=-1)
        for t in range(self.num_tokens):
            decoder_output = self.tf_dec(
                self.positional_encoder(self.dictionary(z_input)),
                slots
            )
            z_next = tf.one_hot(tf.math.argmax(self.out(decoder_output)[:, -1:], axis=-1), depth=self.vocab_size)
            z_gen = z_next if t == 0 else tf.concat((z_gen, z_next), axis=1)
            z_input = tf.concat([
                z_input,
                tf.concat([tf.zeros_like(z_next[:, :, :1]), z_next], axis=-1)
            ], axis=1)

        return z_gen

    @tf.function
    def call(self, z_input, z_target):
        emb_input = self.embed_tokens(z_input)
        slots, attns = self.apply_slot_attn(emb_input)
        pred = self.parallel_decode(emb_input, slots)
        cross_entropy = -tf.reduce_mean(eo.reduce(z_target * tf.nn.log_softmax(pred, axis=-1), '... s d -> ...', 'sum'))
        outputs = {'attns': attns}
        metrics = {'cross_entropy': cross_entropy}
        return outputs, metrics

    @staticmethod
    @tf.function
    def loss_and_grad(slot_model, z_input, z_target):
        with tf.GradientTape() as tape:
            output, metrics = slot_model(z_input, z_target)
        gradients = tape.gradient(metrics['cross_entropy'], slot_model.trainable_weights)
        return output, metrics, gradients

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class DynamicSlotModel(SlotModel):

    @staticmethod
    def defaults_debug():
        debug_args = SlotModel.defaults_debug()
        debug_args.dyn_transformer = transformer.TransformerDecoder.dyn_defaults_debug()
        debug_args.consistency_loss = True
        return debug_args

    @staticmethod
    def defaults():
        default_args = SlotModel.defaults()
        default_args.dyn_transformer = transformer.TransformerDecoder.dyn_defaults()
        default_args.consistency_loss = True
        return default_args


    def __init__(self, vocab_size, num_tokens, args):
        super().__init__(vocab_size, num_tokens, args)

        self.action_encoder = tf.keras.Sequential([
          layers.Dense(args.slot_size, activation='relu'),
          layers.Dense(args.slot_size)
          ])
        self.dynamics = transformer.TransformerDecoder(args.d_model, args.dyn_transformer)

    @staticmethod
    @tf.function
    def loss_and_grad(slot_model, z_input, z_target, action, is_first):
        with tf.GradientTape() as tape:
            output, metrics = slot_model(z_input, z_target, action, is_first)
        gradients = tape.gradient(metrics['loss'], slot_model.trainable_weights)
        return output, metrics, gradients


    @tf.function
    def call(self, z_input, z_target, actions, is_first):
        # TODO: make is_first flag the first action
        # for now, we will manually ignore the first action

        # this requires a flattened input
        emb_input = bottle(self.embed_tokens)(z_input)
        priors, posts, attns = self.filter(slots=None, embeds=emb_input, actions=actions, is_first=is_first)

        # latent loss
        if self.args.consistency_loss:
            consistency = tf.reduce_mean(eo.reduce((priors - posts)**2, 'b t k d -> b t', 'sum'))
        else:
            consistency = tf.cast(tf.convert_to_tensor(0), priors.dtype)

        # loss for both prior and posterior
        concat = lambda x: eo.rearrange(x, 'n b ... -> (n b) ...')
        slots = concat([priors, posts])
        emb_input = concat([emb_input, emb_input])
        z_target = concat([z_target, z_target])

        pred = bottle(self.parallel_decode)(emb_input, slots)
        cross_entropy = -tf.reduce_mean(eo.reduce(z_target * tf.nn.log_softmax(pred, axis=-1), '... s d -> ...', 'sum'))

        outputs = {'attns': attns}  # should later include pred and slots
        metrics = {'cross_entropy': cross_entropy, 'consistency': consistency, 'loss': cross_entropy+consistency}
        return outputs, metrics


    def filter(self, slots, embeds, actions, is_first):
        actions = self.action_encoder(actions)

        priors = []
        posts = []
        attns_seq = []
        post = slots
        for t in range(actions.shape[1]):

            prior, post, attns = self.obs_step(
                prev_state=post, 
                prev_action=actions[:, t], 
                embed=embeds[:, t],
                is_first=is_first[:, t])

            priors.append(prior)
            posts.append(post)
            attns_seq.append(attns)

        priors = eo.rearrange(priors, 't b ... -> b t ...')
        posts = eo.rearrange(posts, 't b ... -> b t ...')
        attns = eo.rearrange(attns_seq, 't b ... -> b t ...')
        # we should also just have a utility function for this
        return priors, posts, attns


    def generate(self, slots, actions):
        actions = self.action_encoder(actions)
        latents = []
        for i in range(actions.shape[1]):
            slots = self.img_step(slots, actions[:, i])
            latents.append(slots)
        latents = rearrange(latents, 't b ... -> b t ...')
        return latents


    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        if prev_state is None:
            prior = self.slot_attn.reset(embed.shape[0])
        else:
            prior = self.img_step(prev_state, prev_action)
        post, attns = self.apply_slot_attn(embed, prior)
        return prior, post, attns


    def img_step(self, prev_state, prev_action, sample=True):
        # prev_action = self.action_encoder(prev_action)
        context = tf.concat([prev_state, rearrange(prev_action, 'b a -> b 1 a')], 1)
        prior = self.dynamics(prev_state, context)
        return prior




"""
loss for posterior only
2021-11-21T19:30:57.044026-0800 [121] kl_loss 0 / image_loss 5557.73 / reward_loss 0 / discount_loss 0 / model_kl 990.78 / prior_ent 0 / post_ent 0 / slate/loss 6548.52 / slate/mse 5557.73 / slate/cross_entropy 990.78 / slate/slot_model_lr 1e-4 / slate/dvae_lr 3e-4 / slate/itr 7 / slate/tau 0.1 / fps 0.18

loss for prior and posterior
2021-11-21T21:40:54.892150-0800 [121] kl_loss 0 / image_loss 5557.73 / reward_loss 0 / discount_loss 0 / model_kl 982.31 / prior_ent 0 / post_ent 0 / slate/loss 6540.05 / slate/mse 5557.73 / slate/cross_entropy 982.31 / slate/slot_model_lr 1e-4 / slate/dvae_lr 3e-4 / slate/itr 7 / slate/tau 0.1 / fps 0.19



"""


