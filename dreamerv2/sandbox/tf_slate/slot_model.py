from utils import *
import slot_attn
# import slot_heads
import transformer

import einops as eo
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.mixed_precision import experimental as prec


# dreamer stuff
# from common import tfutils
from sandbox import machine

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

class EinsumOneHotDictionary(layers.Layer):
    """
        first test whether doing this will allow gradient to flow through the input
        if it does, then just use a weight matrix instead of an embedding layer
    """
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.dictionary = layers.Embedding(vocab_size, emb_size)
        # self.dictionary.variables[0].shape: (vocab_size, emb_size)

    def call(self, x):
        """
        x: B, N, vocab_size
        """
        tokens = tf.math.argmax(x, axis=-1)
        dummy_output = self.dictionary(tokens)  # batch_size x N x emb_size
        # onehots = tf.one_hot(tokens, depth=self.vocab_size, axis=1)  # (b, vocab_size, N)
        token_embs = tf.einsum('bnv,vd->bnd', x, self.dictionary.variables[0])  # batch_size x N x emb_size
        return token_embs


class SlotHead(tkl.Layer):
    @staticmethod
    def defaults_debug():
        debug_args = SlotHead.defaults()
        debug_args.head = transformer.TransformerDecoder.head_defaults_debug()
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            head=transformer.TransformerDecoder.small_head_defaults()
            ))
        return default_args

    def __init__(self, slot_size, out_size, cfg):
        super().__init__()
        self.head = transformer.TransformerDecoder(slot_size, cfg.head)
        self.out = linear(slot_size, out_size)


    def call(self, slots):
        """
        x: B, K, D
        """
        x = self.head(slots, slots)
        x = eo.reduce(x, 'b k d -> b d', 'mean')
        out = self.out(x)
        return out

class DistSlotHead(tkl.Layer):
    @staticmethod
    def defaults_debug():
        debug_args = DistSlotHead.defaults()
        debug_args.head = transformer.TransformerDecoder.head_defaults_debug()
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            head=transformer.TransformerDecoder.small_head_defaults()
            ))
        return default_args

    def __init__(self, slot_size, shape, dist_cfg, cfg):
        super().__init__()
        self.slot_size = slot_size
        self._shape = (shape,) if isinstance(shape, int) else shape
        self.head = transformer.TransformerDecoder(slot_size, cfg.head)
        self.out = machine.DistLayer(self._shape, **dist_cfg)

    def call(self, slots):
        """
        x: (B K D) or (H B K D)

        TODO: get rid of the hacky case analysis. Essentially you want to be able to bottle if necessary, otherwise don't bottle.
        """
        # if len(slots.shape) == 2:
        #     num_slots = 1
        #     slots = eo.rearrange(x, '... (k d) -> ... k d', k=num_slots)
        if len(slots.shape) == 3:
            x = self.head(slots, slots)
        elif len(slots.shape) == 4:
            x = bottle(self.head)(slots, slots)
        else:
            raise NotImplementedError
        x = eo.reduce(x, '... k d -> ... d', 'mean')
        out = self.out(x)
        return out


class SelfAttnHead(DistSlotHead):

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            head=transformer.TransformerDecoder.two_blocks_eight_heads_defaults()
            ))
        return default_args

    def __init__(self, slot_size, shape, dist_cfg, cfg):
        DistSlotHead.__init__(self, slot_size, shape, dist_cfg, cfg)
        self.encoder = layers.Dense(slot_size)

    def call(self, slots):
        """
        x: (B K D) or (H B K D)

        TODO: get rid of the hacky case analysis. Essentially you want to be able to bottle if necessary, otherwise don't bottle.
        """
        num_slots = 1
        slots = eo.rearrange(slots, '... (k d) -> ... k d', k=num_slots)

        slots = self.encoder(slots)
        if len(slots.shape) == 3:
            x = self.head(slots, slots)
        elif len(slots.shape) == 4:
            x = bottle(self.head)(slots, slots)
        else:
            raise NotImplementedError
        x = eo.reduce(x, '... k d -> ... d', 'mean')
        out = self.out(x)
        return out


class CrossAttnHead(DistSlotHead):

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            head=transformer.TransformerDecoder.two_blocks_eight_heads_defaults()
            ))
        return default_args

    def __init__(self, slot_size, shape, dist_cfg, cfg):
        DistSlotHead.__init__(self, slot_size, shape, dist_cfg, cfg)
        self.encoder = layers.Dense(slot_size)
        # self.query = tf.Variable(tf.random.truncated_normal(
        #     (1,) if self._shape == [] else self._shape))
        self.query = tf.Variable(tf.random.truncated_normal((slot_size,)))

    def call(self, slots):
        """
        x: (B K D) or (H B K D)

        TODO: get rid of the hacky case analysis. Essentially you want to be able to bottle if necessary, otherwise don't bottle.
        """
        slots = self.encoder(slots)
        if len(slots.shape) == 3:
            query = eo.repeat(self.query, 'd -> b 1 d', b=slots.shape[0])
            x = self.head(query, slots)
        elif len(slots.shape) == 4:
            query = eo.repeat(self.query, 'd -> b t 1 d', b=slots.shape[0], t=slots.shape[1])
            x = bottle(self.head)(query, slots)
        else:
            raise NotImplementedError
        x = eo.rearrange(x, '... q d -> ... (q d)')  # note q = 1. Although you could use the same transformer network for multiple heads I suppose
        out = self.out(x)
        return out


class SlotModel(layers.Layer):

    @staticmethod
    def defaults_debug():
        debug_args = SlotModel.defaults()
        debug_args.d_model = 16
        debug_args.slot_size = 16
        debug_args.lr_warmup_steps = 3
        debug_args.obs_transformer = transformer.TransformerDecoder.obs_defaults_debug()
        debug_args.slot_attn=slot_attn.SlotAttentionWithReset.defaults_debug()
        debug_args.einsum_dict=False
        debug_args.perceiver_output=False
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            lr=1e-4,
            lr_warmup_steps=30000,

            d_model=192,
            dropout=0.1,

            obs_transformer=transformer.TransformerDecoder.obs_defaults(),
            obs_cross_transformer=transformer.TransformerDecoder.obs_cross_defaults(),

            slot_attn=slot_attn.SlotAttentionWithReset.defaults(),
            slot_size=192,

            einsum_dict=True,

            decoder_type='tfdec',  # or crsatn
            perceiver_output=False,
            ))
        return default_args

    def __init__(self, vocab_size, num_tokens, args):
        super().__init__()

        self.args = args
        self.vocab_size = vocab_size
        self.d_model = args.d_model
        self.num_tokens = num_tokens
        self.perceiver_output = args.perceiver_output

        # obs encoder
        if self.args.einsum_dict:
            self.dictionary = EinsumOneHotDictionary(self.vocab_size + 1, args.d_model)
        else:
            self.dictionary = OneHotDictionary(self.vocab_size + 1, args.d_model)
        self.positional_encoder = transformer.PositionalEncoding(1 + self.num_tokens, args.d_model, args.dropout)
        self.layer_norm = tkl.LayerNormalization(epsilon=1e-5)
        self.token_mlp = tf.keras.Sequential([
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            tkl.ReLU(),
            linear(args.d_model, args.d_model)])

        # recurrent: replace this with rssm
        self.slot_attn = slot_attn.SlotAttentionWithReset(args.slot_size, args.slot_attn)
        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)

        # decoder
        if self.perceiver_output:
            # assume that resolution is square
            res = int(self.num_tokens**0.5)
            assert res**2 == self.num_tokens
            self._resolution = (res, res)
            self.output_position_encoding = transformer.CoordConvPositionalEncoding(resolution=self._resolution, dim=args.d_model)
            self.output_token_mlp = tf.keras.Sequential([
                linear(args.d_model, args.d_model, weight_init='kaiming'),
                tkl.ReLU(),
                linear(args.d_model, args.d_model)])

        if args.decoder_type == 'tfdec':
            self.tf_dec = transformer.TransformerDecoder(args.d_model, args.obs_transformer)
        elif args.decoder_type == 'crsatn':
            self.tf_dec = transformer.CrossAttentionStack(args.d_model, args.obs_cross_transformer)
        else:
            raise NotImplementedError
        self.out = linear(args.d_model, self.vocab_size, bias=False)

        self.training = False

    def initial(self, batch_size):
        return self.slot_attn.reset(batch_size)

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
            z_next = self.logits_to_tokens(self.out(decoder_output)[:, -1:])
            z_gen = z_next if t == 0 else tf.concat((z_gen, z_next), axis=1)
            z_input = tf.concat([
                z_input,
                tf.concat([tf.zeros_like(z_next[:, :, :1]), z_next], axis=-1)
            ], axis=1)
        return z_gen

    def logits_to_tokens(self, logits):
        """
            (B, sequence_length, vocab_size)
        """
        return tf.one_hot(tf.math.argmax(logits, axis=-1), depth=self.vocab_size)

    def perceiver_decode(self, slots):
        """
        This would replace parallel_decode in training and autoregressive decode in evaluation
        """
        # create coordconv position embeddings
        bsize = slots.shape[0]
        queries = tf.zeros([bsize] + list(self._resolution) + [self.d_model], dtype=slots.dtype)
        queries = self.output_token_mlp(self.output_position_encoding(queries))
        queries = eo.rearrange(queries, '... h w d -> ... (h w) d')

        # tf decode
        decoder_output = self.tf_dec(queries, slots)
        pred = self.out(decoder_output)  # these are what goes into the log softmax
        return pred

    @staticmethod
    def cross_entropy_loss(pred, target):
        return -tf.reduce_mean(eo.reduce(target * tf.nn.log_softmax(pred, axis=-1), '... s d -> ...', 'sum'))

    @tf.function
    def call(self, z_input, z_target, embeds):
        emb_input = self.embed_tokens(z_input)
        slots, attns = self.apply_slot_attn(emb_input)
        if self.perceiver_output:
            pred = self.perceiver_decode(slots)
        else:
            pred = self.parallel_decode(emb_input, slots)
        cross_entropy = SlotModel.cross_entropy_loss(pred, z_target)
        outputs = {'attns': attns, 'slots': slots}
        metrics = {'cross_entropy': cross_entropy}
        loss = cross_entropy
        return loss, outputs, metrics

    @tf.function
    def loss_and_grad(self, z_input, z_target, embeds):
        with tf.GradientTape() as tape:
            loss, output, metrics = self(z_input, z_target, embeds)
        gradients = tape.gradient(loss, self.trainable_weights)
        return loss, output, metrics, gradients

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class DynamicSlotModel(SlotModel, machine.EnsembleRSSM):
    # will search SlotModel first, then machine.EnsembleRSSM

    """
        what args are relevant for:
            - kl_loss
                - forward
                - balance
                - free
                - free_avg
            - _suff_stats_layer
                - self._discrete
                - self._stoch
                - self._std_act
                - self._min_std
            - _suff_stats_ensemble
                - self._hidden
                - self._norm
                - self._ensemble
            - get_dist
                - self._discrete
            - DistLayer
                - min_std
                - dist

        I guess I could just use self.global_config.rssm for the relevant args though.

    """

    @staticmethod
    def defaults_debug():
        debug_args = SlotModel.defaults_debug()
        debug_args.dyn_transformer = transformer.TransformerDecoder.dyn_defaults_debug()
        debug_args.consistency_loss = True
        debug_args.lr = 3e-4
        debug_args.min_lr_factor = 0.2
        debug_args.decay_steps = 15
        debug_args.reward_head = SlotHead.defaults_debug()
        debug_args.distributional = False
        return debug_args

    @staticmethod
    def defaults():
        default_args = SlotModel.defaults()
        default_args.dyn_transformer = transformer.TransformerDecoder.dyn_defaults()
        default_args.slot_attn = slot_attn.SlotAttentionWithReset.dyn_defaults()
        default_args.consistency_loss = True
        default_args.lr = 3e-4
        default_args.min_lr_factor = 0.2
        default_args.decay_steps = 30000
        default_args.reward_head = DistSlotHead.defaults()
        default_args.distributional = False
        return default_args


    def __init__(self, vocab_size, num_tokens, args, global_config):
        SlotModel.__init__(self, vocab_size, num_tokens, args)
        self.global_config = global_config  # a hack that we will remove once we integrate with RSSM

        if self.args.distributional:
            self._ensemble = self.global_config.rssm.ensemble
            self._stoch = self.global_config.rssm.stoch
            self._deter = self.global_config.rssm.deter
            self._hidden = self.global_config.rssm.hidden
            self._discrete = self.global_config.rssm.discrete
            self._act = machine.get_act(self.global_config.rssm.act)
            self._norm = self.global_config.rssm.norm
            self._std_act = self.global_config.rssm.std_act
            self._min_std = self.global_config.rssm.min_std

        self.action_encoder = tf.keras.Sequential([
          layers.Dense(args.slot_size, activation='relu'),
          layers.Dense(args.slot_size)
          ])
        self.dynamics = transformer.TransformerDecoder(args.d_model, args.dyn_transformer)

        # other heads
        self.heads = {}
        self.heads['reward'] = DistSlotHead(
            slot_size=args.slot_size, 
            shape=[],
            dist_cfg=dict(dist=self.global_config.reward_head.dist),
            cfg=args.reward_head)


    @tf.function
    def loss_and_grad(self, z_input, z_target, action, is_first, reward, embeds):
        with tf.GradientTape() as tape:
            loss, output, metrics = self(z_input, z_target, action, is_first, reward)
        gradients = tape.gradient(loss, self.trainable_weights)
        return loss, output, metrics, gradients


    @tf.function
    def call(self, z_input, z_target, action, is_first, reward, embeds):
        # TODO: make is_first flag the first action
        # for now, we will manually ignore the first action

        emb_input = bottle(self.embed_tokens)(z_input)
        priors, posts, attns = self.filter(slots=None, embeds=emb_input, actions=action, is_first=is_first)

        # # HACK
        # priors = priors['deter']
        # posts = posts['deter']

        # latent loss
        if self.args.consistency_loss:
            consistency = tf.reduce_mean(eo.reduce((priors['deter'] - posts['deter'])**2, 'b t k d -> b t k', 'sum'))
        else:
            consistency = tf.cast(tf.convert_to_tensor(0), priors['deter'].dtype)

        # loss for both prior and posterior
        concat = lambda x: eo.rearrange(x, 'n b ... -> (n b) ...')
        slots = concat([priors['deter'], posts['deter']])
        emb_input = concat([emb_input, emb_input])
        z_target = concat([z_target, z_target])
        rew_target = concat([reward, reward])

        if self.perceiver_output:
            pred = bottle(self.perceiver_decode)(slots)
        else:
            pred = bottle(self.parallel_decode)(emb_input, slots)
        cross_entropy = SlotModel.cross_entropy_loss(pred, z_target)

        ###########################################################
        feat = slots
        likes = {}
        losses = {}
        data = {'reward': rew_target}
        # *********************************************************
        # copied from WorldModel.loss()
        for name, head in self.heads.items():
          grad_head = (name in self.global_config.grad_heads)
          inp = feat if grad_head else tf.stop_gradient(feat)
          out = head(inp)
          dists = out if isinstance(out, dict) else {name: out}
          for key, dist in dists.items():
            like = tf.cast(dist.log_prob(data[key]), tf.float32)
            likes[key] = like
            losses[key] = -like.mean()
        # *********************************************************
        rew_loss = self.global_config.loss_scales.get('reward', 1.0) * losses['reward'] 
        ###########################################################
        outputs = {'attns': attns, 'post': posts, 'prior': priors, 'pred': pred}
        metrics = {'cross_entropy': cross_entropy, 'consistency': consistency, 'rew_loss': rew_loss}
        loss = tf.reduce_sum([
            cross_entropy, 
            consistency,
            rew_loss
            ])
        return loss, outputs, metrics


    def get_feat(self, slots):
        # HACK for now, but later you will make your latent a dictionary
        if isinstance(slots, dict):
            return slots['deter']
        else:
            return slots


    def filter(self, slots, embeds, actions, is_first):
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

        swap = lambda x: eo.rearrange(x, 't b ... -> b t ...')
        priors = {k: tf.stack([prior[k] for prior in priors], axis=1) for k in priors[0].keys()}
        posts = {k: tf.stack([post[k] for post in posts], axis=1) for k in posts[0].keys()}
        attns = eo.rearrange(attns_seq, 't b ... -> b t ...')

        # this can probably be achieved through the static scan

        # we should also just have a utility function for this
        return priors, posts, attns


    def generate(self, slots, actions):
        latents = []
        for i in range(actions.shape[1]):
            slots = self.img_step(slots, actions[:, i])
            latents.append(slots)
        latents = {k: tf.stack([latent[k] for latent in latents], axis=1) for k in latents[0].keys()}
        return latents

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self.args.distributional:
            state = dict(
              logit=tf.zeros([batch_size, self.slot_attn.num_slots, self._stoch, self._discrete], dtype),
              stoch=tf.zeros([batch_size, self.slot_attn.num_slots, self._stoch, self._discrete], dtype),
              deter=self.slot_attn.reset(batch_size)
                )
        else:
            state = dict(deter=self.slot_attn.reset(batch_size))
        return state


    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):

        if prev_state is None:
            prior = self.initial(embed.shape[0])
        else:
            resetted_prior = self.initial(embed.shape[0])
            predicted_prior = self.img_step(prev_state, prev_action)
            # mask = rearrange(is_first.astype(prev_state['deter'].dtype), 'b -> b 1 1')
            # prior = mask * resetted_prior['deter'] + (1 - mask) * predicted_prior['deter']

            # mask should be b -> b 1 1 for deter
            # mask should be b -> b 1 1 1 for stoch and logits
            # prior = {key: mask * resetted_prior[key] + (1 - mask) * predicted_prior[key] for key in predicted_prior}

            prior = {}
            for key in predicted_prior:
                if key == 'deter':
                    mask = rearrange(is_first.astype(prev_state['deter'].dtype), 'b -> b 1 1')
                elif key in ['stoch', 'logit']:
                    mask = rearrange(is_first.astype(prev_state['deter'].dtype), 'b -> b 1 1 1')
                else:
                    raise NotImplementedError
                prior[key] = mask * resetted_prior[key] + (1 - mask) * predicted_prior[key]

        post, attns = self.apply_slot_attn(embed, prior['deter'])  # TODO or should you do get feat?
        if self.args.distributional:
            # import ipdb
            x = post
            ########################################################
            # copied directly from machine
            stats = self._suff_stats_layer('obs_dist', x)
            dist = self.get_dist(stats)
            stoch = dist.sample() if sample else dist.mode()
            post = {'stoch': stoch, 'deter': prior['deter'], **stats}
            ########################################################
        else:
            post = {'deter': post}

        return prior, post, attns


    def img_step(self, prev_state, prev_action, sample=True):
        """
            prev_state: {deter: (B, K, D)}
            prev_action: (B, A)
        """
        prev_state = prev_state['deter']
        prev_action = self.action_encoder(prev_action)
        context = tf.concat([prev_state, rearrange(prev_action, 'b a -> b 1 a')], 1)
        deter = self.dynamics(prev_state, context)

        if self.args.distributional:
            # import ipdb
            x = deter
            ########################################################
            # copied directly from machine
            stats = self._suff_stats_ensemble(x)
            index = tf.random.uniform((), 0, self._ensemble, tf.int32)
            stats = {k: v[index] for k, v in stats.items()}
            dist = self.get_dist(stats)
            stoch = dist.sample() if sample else dist.mode()
            prior = {'stoch': stoch, 'deter': deter, **stats}
            ########################################################
        else:
            prior = {'deter': deter}

        return prior


    def imag_autoregressive(self, slots, actions):
        """
            slots: (b, k, ds)
            actions: (b, t, da)
        """
        imag_latent = self.generate(slots, actions)
        if self.perceiver_output:
            pred = bottle(self.perceiver_decode)(self.get_feat(imag_latent))
            z_gen = self.logits_to_tokens(pred)
        else:
            z_gen = bottle(self.autoregressive_decode)(self.get_feat(imag_latent))
        output = {'z_gen': z_gen}
        return output

    def recon_autoregressive(self, z_input, actions, is_first):
        """
            image: TensorShape([6, 5, 64, 64, 3])
            actions: TensorShape([6, 5, 9])
            is_first: TensorShape([6,5])
        """
        emb_input = bottle(self.embed_tokens)(z_input)
        priors, posts, attns = self.filter(slots=None, embeds=emb_input, actions=actions, is_first=is_first)
        if self.perceiver_output:
            pred = bottle(self.perceiver_decode)(self.get_feat(posts))
            z_gen = self.logits_to_tokens(pred)
        else:
            z_gen = bottle(self.autoregressive_decode)(self.get_feat(posts))
        output = {'slots': posts, 'attns': attns, 'z_gen': z_gen}
        return output


    # or maybe I can put imagine and reconstruct here, and then in slate I just have one rollout function that just decodes everything all at once? 



"""
loss for posterior only
2021-11-21T19:30:57.044026-0800 [121] kl_loss 0 / image_loss 5557.73 / reward_loss 0 / discount_loss 0 / model_kl 990.78 / prior_ent 0 / post_ent 0 / slate/loss 6548.52 / slate/mse 5557.73 / slate/cross_entropy 990.78 / slate/slot_model_lr 1e-4 / slate/dvae_lr 3e-4 / slate/itr 7 / slate/tau 0.1 / fps 0.18

loss for prior and posterior
2021-11-21T21:40:54.892150-0800 [121] kl_loss 0 / image_loss 5557.73 / reward_loss 0 / discount_loss 0 / model_kl 982.31 / prior_ent 0 / post_ent 0 / slate/loss 6540.05 / slate/mse 5557.73 / slate/cross_entropy 982.31 / slate/slot_model_lr 1e-4 / slate/dvae_lr 3e-4 / slate/itr 7 / slate/tau 0.1 / fps 0.19



"""


