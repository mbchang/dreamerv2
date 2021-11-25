from utils import *
import dvae
import slot_model

from einops import rearrange, repeat
from loguru import logger as lgr
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as layers
import time
import wandb


def create_tokens(z_hard):
    # target tokens for transformer
    z_target = rearrange(z_hard, 'b c h w -> b (h w) c')

    # add BOS token
    B, zhw, zc = z_target.shape
    z_input = tf.concat([tf.zeros((B, zhw, 1)), z_target], axis=-1)
    z_input = tf.concat([
        tf.concat([tf.ones((B, 1, 1)), tf.zeros((B, 1, zc))], axis=-1),
         z_input], axis=-2)
    return z_input, z_target


def overlay_attention(attns, image):
    *_, H, W = image.shape
    attns = rearrange(attns, 'b hw k -> b k hw')
    size = int(np.sqrt(attns.shape[-1]))
    attns = tf.repeat(tf.repeat(rearrange(attns, 'b k (h w) -> b k 1 h w', h = size, w=size), H // size, axis=-2), W // size, axis=-1)
    attns = image * attns + 1. - attns
    return attns


class SLATE(layers.Layer):

    @staticmethod
    def defaults_debug():
        debug_args = SLATE.defaults()
        debug_args.log_interval = 8
        debug_args.batch_size = 5
        debug_args.vocab_size = 32
        debug_args.dvae = dvae.dVAE.defaults_debug()
        debug_args.slot_model = slot_model.SlotModel.defaults_debug()
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            log_interval=800,

            image_size=64,
            img_channels=3,

            batch_size=50,
            lr_decay_factor=1.0,

            vocab_size=1024,
            dvae=dvae.dVAE.defaults(),
            slot_model=slot_model.SlotModel.defaults(),

            mono_train=False,
            # clip=1.0,
            ))
        return default_args

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_tokens = (args.image_size // 4) ** 2

        self.dvae = dvae.dVAE(args.vocab_size, args.img_channels)
        self.slot_model = slot_model.SlotModel(args.vocab_size, self.num_tokens, args.slot_model)

        self.configure_optimizers(self.args)

        self.training = False
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def configure_optimizers(self, args):
        self.dvae_optimizer = tf.keras.optimizers.Adam(args.dvae.lr, epsilon=1e-08)
        self.main_optimizer = tf.keras.optimizers.Adam(args.slot_model.lr, epsilon=1e-08)
        # if self.args.mono_train:
        #     # initialize with fake input
        #     self(tf.random.uniform((1, 3, 64, 64)), tf.constant(1.0), True)  
        #     self.optimizer = tfa.optimizers.MultiOptimizer([
        #         (self.dvae_optimizer, self.dvae),
        #         (self.main_optimizer, self.slot_model)
        #     ])#, clipvalue=args.clip)

    def call(self, image: tf.Tensor, tau: tf.Tensor, hard: bool):
        """
        image: batch_size x img_channels x H x W
        """
        dvae_loss, dvae_out, dvae_mets = self.dvae(image, tau, hard)

        z_input, z_target = create_tokens(tf.stop_gradient(dvae_out['z_hard']))
        sm_loss, sm_out, sm_mets = self.slot_model(z_input, z_target)

        outputs = dict(dvae=dvae_out, slot_model=sm_out)
        metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
        loss = dvae_loss + sm_loss
        return loss, outputs, metrics

    def decode(self, z):
        size = int(np.sqrt(self.num_tokens))
        z = tf.cast(rearrange(z, 'b (h w) d -> b d h w', h=size, w=size), tf.float32)
        output = self.dvae.decoder(z)
        return output

    def image_to_argmax_tokens(self, image):
        z_logits = self.dvae.get_logits(image)
        z_hard = self.dvae.mode(z_logits)
        one_hot_tokens, _ = create_tokens(z_hard)
        return one_hot_tokens

    @tf.function
    def reconstruct_autoregressive(self, image: tf.Tensor, eval: bool=False):
        """
        image: batch_size x img_channels x H x W
        """
        one_hot_tokens = self.image_to_argmax_tokens(image)
        emb_input = self.slot_model.embed_tokens(one_hot_tokens)
        slots, attns = self.slot_model.apply_slot_attn(emb_input)
        z_gen = self.slot_model.autoregressive_decode(slots)
        recon_transformer = self.decode(z_gen)
        return recon_transformer, slots, attns

    def get_iterates(self, step):
        tau = cosine_anneal(
            step=step,
            start_value=self.args.dvae.tau_start,
            final_value=self.args.dvae.tau_final,
            start_step=0,
            final_step=self.args.dvae.tau_steps)

        lr_warmup_factor = linear_warmup(
            step=step,
            start_value=0.,
            final_value=1.0,
            start_step=0,
            final_step=self.args.slot_model.lr_warmup_steps)

        return dict(tau=tau, lr_warmup_factor=lr_warmup_factor)

    def train_step(self, image):
        iterates = self.get_iterates(self.step.numpy())

        self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
        self.main_optimizer.lr = f32(self.args.lr_decay_factor * iterates['lr_warmup_factor'] * self.args.slot_model.lr)

        dvae_loss, dvae_out, dvae_mets, gradients = self.dvae.loss_and_grad(image, tf.constant(iterates['tau']), self.args.dvae.hard)
        self.dvae_optimizer.apply_gradients(zip(gradients, self.dvae.trainable_weights))

        z_input, z_target = create_tokens(tf.stop_gradient(dvae_out['z_hard']))

        sm_loss, sm_out, sm_mets, gradients = self.slot_model.loss_and_grad(z_input, z_target)
        # NOTE: if we put this inside tf.function then the performance becomes very bad
        self.main_optimizer.apply_gradients(zip(gradients, self.slot_model.trainable_weights))

        loss = dvae_loss + sm_loss

        outputs = dict(dvae=dvae_out, slot_model=sm_out, iterates=iterates)
        metrics = dict(dvae=dvae_mets, slot_model=sm_mets)

        self.step.assign_add(1)

        return loss, outputs, metrics

    @tf.function
    def loss_and_grad(self, image, tau, hard):
        with tf.GradientTape() as dvae_tape, tf.GradientTape() as sm_tape:
            loss, outputs, metrics = self(image, tau, hard)

        dvae_grads = dvae_tape.gradient(loss, self.dvae.trainable_weights)
        sm_grads = sm_tape.gradient(loss, self.slot_model.trainable_weights)

        gradients = {'dvae': dvae_grads, 'slot_model': sm_grads}
        return loss, outputs, metrics, gradients


    def monolithic_train_step(self, image):
        iterates = self.get_iterates(self.step.numpy())

        self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
        self.main_optimizer.lr = f32(self.args.lr_decay_factor * iterates['lr_warmup_factor'] * self.args.slot_model.lr)

        loss, outputs, metrics, gradients = self.loss_and_grad(image, tf.constant(iterates['tau']), self.args.dvae.hard)

        self.dvae_optimizer.apply_gradients(zip(gradients['dvae'], self.dvae.trainable_weights))
        self.main_optimizer.apply_gradients(zip(gradients['slot_model'], self.slot_model.trainable_weights))

        outputs = {'iterates': iterates, **outputs}

        self.step.assign_add(1)

        return loss, outputs, metrics


    # this should either have the teaching-forcing mode or the autoregressive mode
    @staticmethod
    def visualize(image, attns, recon, gen_img, preproc):
        unsqueeze = lambda x: rearrange(preproc(x), 'b c h w -> b 1 c h w')
        vis_recon = tf.concat((
            unsqueeze(image), 
            unsqueeze(recon), 
            unsqueeze(gen_img), 
            overlay_attention(attns, unsqueeze(image))), axis=1)
        return vis_recon

    def train(self):
        self.training = True
        self.slot_model.train()

    def eval(self):
        self.training = False
        self.slot_model.eval()


class DynamicSLATE(SLATE):

    @staticmethod
    def defaults_debug():
        debug_args = SLATE.defaults_debug()
        debug_args.slot_model = slot_model.DynamicSlotModel.defaults_debug()
        debug_args.vis_rollout = True
        return debug_args

    @staticmethod
    def defaults():
        default_args = SLATE.defaults()
        default_args.slot_model = slot_model.DynamicSlotModel.defaults()
        default_args.vis_rollout = True
        return default_args

    def __init__(self, args):
        layers.Layer.__init__(self)
        self.args = args

        self.dvae = dvae.dVAE(args.vocab_size, args.img_channels)
        self.dvae_optimizer = tf.keras.optimizers.Adam(args.dvae.lr, epsilon=1e-08)

        self.num_tokens = (args.image_size // 4) ** 2
        self.slot_model = slot_model.DynamicSlotModel(args.vocab_size, self.num_tokens, args.slot_model)
        self.main_optimizer = tf.keras.optimizers.Adam(args.slot_model.lr, epsilon=1e-08)

        self.training = False
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def call(self, data, tau, hard):
        """
        image: batch_size x img_channels x H x W
        """
        permute = lambda x: rearrange(x, '... h w c -> ... c h w')
        flatten = lambda x: rearrange(x, 'b t ... -> (b t) ...')
        unflatten = lambda x: rearrange(x, '(b t) ... -> b t ...', b=data['action'].shape[0])

        dvae_loss, dvae_out, dvae_mets = self.dvae(flatten(permute(data['image'])), tau, hard)
        z_input, z_target = create_tokens(tf.stop_gradient(dvae_out['z_hard']))
        sm_loss, sm_out, sm_mets = self.slot_model(unflatten(z_input), unflatten(z_target), data['action'], data['is_first'])

        sm_out['attns'] = flatten(sm_out['attns'])

        outputs = dict(dvae=dvae_out, slot_model=sm_out)
        metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
        loss = dvae_loss + sm_loss
        return loss, outputs, metrics

    def imagine(self, slots, actions):
        """
            slots: (b, k, ds)
            actions: (b, t, da)
        """
        imag_latent = self.slot_model.generate(slots, actions)
        z_gen = bottle(self.slot_model.autoregressive_decode)(imag_latent)
        recon_transformer = bottle(self.decode)(z_gen)
        output = {'pred': recon_transformer}
        metrics = {}  # will later have cross entropy and mse
        return output, metrics

    def reconstruct(self, data):
        """
            image: TensorShape([6, 5, 64, 64, 3])
            actions: TensorShape([6, 5, 9])
            is_first: TensorShape([6,5])
        """
        permute = lambda x: rearrange(x, '... h w c -> ... c h w')
        flatten = lambda x: rearrange(x, 'b t ... -> (b t) ...')
        unflatten = lambda x: rearrange(x, '(b t) ... -> b t ...', b=data['action'].shape[0])

        image = flatten(permute(data['image']))
        one_hot_tokens = unflatten(self.image_to_argmax_tokens(image))
        emb_input = bottle(self.slot_model.embed_tokens)(one_hot_tokens)
        priors, posts, attns = self.slot_model.filter(slots=None, embeds=emb_input, actions=data['action'], is_first=data['is_first'])
        z_gen = bottle(self.slot_model.autoregressive_decode)(posts)
        recon_transformer = bottle(self.decode)(z_gen)
        output = {'pred': recon_transformer, 'slots': posts, 'attns': attns}
        metrics = {}  # will later have cross entropy and mse
        return output, metrics

    def rollout(self, batch, seed_steps, pred_horizon):
        batch_horizon = tf.nest.map_structure(lambda x: x[:, :seed_steps + pred_horizon], batch)
        batch_seed = tf.nest.map_structure(lambda x: x[:, :seed_steps], batch)

        # this could actually be done via parallel decode I suppose
        recon_output, recon_metrics = self.reconstruct(batch_seed)  
        if pred_horizon > 0:
            imag_output, imag_metrics = self.imagine(recon_output['slots'][:, -1], batch_horizon['action'][:, seed_steps:])
            output = {'video': tf.concat((recon_output['pred'], imag_output['pred']), axis=1)}
            metrics = {**recon_metrics, **imag_metrics}
        else:
            output = {'video': recon_output['pred']}
            metrics = recon_metrics
        return output, metrics


    def train_step(self, data):
        """
          reward (B, T)
          is_first (B, T)
          is_last (B, T)
          is_terminal (B, T)
          image (B, T, H, W, C)
          orientations (B, T, D)
          height (B, T)
          velocity (B, T, V)
          action (B, T, A)
        """
        permute = lambda x: rearrange(x, '... h w c -> ... c h w')
        flatten = lambda x: rearrange(x, 'b t ... -> (b t) ...')
        unflatten = lambda x: rearrange(x, '(b t) ... -> b t ...', b=data['action'].shape[0])

        iterates = self.get_iterates(self.step.numpy())

        self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
        self.main_optimizer.lr = f32(self.args.lr_decay_factor * iterates['lr_warmup_factor'] * self.args.slot_model.lr)

        dvae_loss, dvae_out, dvae_mets, gradients = self.dvae.loss_and_grad(flatten(permute(data['image'])), tf.constant(iterates['tau']), self.args.dvae.hard)
        self.dvae_optimizer.apply_gradients(zip(gradients, self.dvae.trainable_weights))

        z_input, z_target = create_tokens(tf.stop_gradient(dvae_out['z_hard']))

        sm_loss, sm_out, sm_mets, gradients = self.slot_model.loss_and_grad(
            unflatten(z_input),
            unflatten(z_target),
            action=data['action'],
            is_first=data['is_first']
            )
        # NOTE: if we put this inside tf.function then the performance becomes very bad
        self.main_optimizer.apply_gradients(zip(gradients, self.slot_model.trainable_weights))

        loss = dvae_loss + sm_loss

        outputs = dict(dvae=dvae_out, slot_model=sm_out, iterates=iterates)
        metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
        self.step.assign_add(1)

        return loss, outputs, metrics
