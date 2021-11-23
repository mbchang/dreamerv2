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
    z_transformer_target = rearrange(z_hard, 'b c h w -> b (h w) c')

    # add BOS token
    B, zhw, zc = z_transformer_target.shape
    z_transformer_input = tf.concat([tf.zeros((B, zhw, 1)), z_transformer_target], axis=-1)
    z_transformer_input = tf.concat([
        tf.concat([tf.ones((B, 1, 1)), tf.zeros((B, 1, zc))], axis=-1),
         z_transformer_input], axis=-2)
    return z_transformer_input, z_transformer_target


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
            ))
        return default_args

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dvae = dvae.dVAE(args.vocab_size, args.img_channels)
        self.dvae_optimizer = tf.keras.optimizers.Adam(args.dvae.lr, epsilon=1e-08)

        self.num_tokens = (args.image_size // 4) ** 2
        self.slot_model = slot_model.SlotModel(args.vocab_size, self.num_tokens, args.slot_model)
        self.main_optimizer = tf.keras.optimizers.Adam(args.slot_model.lr, epsilon=1e-08)

        self.training = False
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def call(self, image: tf.Tensor, tau: tf.Tensor, hard: bool):
        """
        image: batch_size x img_channels x H x W
        """
        dvae_out, dvae_mets = self.dvae(image, tau, hard)

        z_transformer_input, z_transformer_target = create_tokens(tf.stop_gradient(dvae_out['z_hard']))
        sm_out, sm_mets = self.slot_model(z_transformer_input, z_transformer_target)

        return (
            dvae_out['recon'],
            sm_mets['cross_entropy'],
            dvae_mets['mse'],
            sm_out['attns'],
            dvae_out['z_hard']
        )

    def decode(self, z):
        size = int(np.sqrt(self.num_tokens))
        z = tf.cast(rearrange(z, 'b (h w) d -> b d h w', h=size, w=size), tf.float32)
        output = self.dvae.decoder(z)
        return output

    @tf.function
    def reconstruct_autoregressive(self, image: tf.Tensor, eval: bool=False):
        """
        image: batch_size x img_channels x H x W
        """
        z_logits = self.dvae.get_logits(image)
        z_hard = self.dvae.mode(z_logits)
        one_hot_tokens, _ = create_tokens(z_hard)
        emb_input = self.slot_model.embed_tokens(one_hot_tokens)
        slots, attns = self.slot_model.apply_slot_attn(emb_input)
        z_gen = self.slot_model.autoregressive_decode(slots)
        recon_transformer = self.decode(z_gen)
        return recon_transformer, slots, attns

    def train_step(self, image):
        # global_step should be the same as self.step

        tau = cosine_anneal(
            step=self.step.numpy(),
            start_value=self.args.dvae.tau_start,
            final_value=self.args.dvae.tau_final,
            start_step=0,
            final_step=self.args.dvae.tau_steps)

        lr_warmup_factor = linear_warmup(
            step=self.step.numpy(),
            start_value=0.,
            final_value=1.0,
            start_step=0,
            final_step=self.args.slot_model.lr_warmup_steps)

        self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
        self.main_optimizer.lr = f32(self.args.lr_decay_factor * lr_warmup_factor * self.args.slot_model.lr)

        dvae_out, dvae_mets, gradients = dvae.dVAE.loss_and_grad(self.dvae, image, tf.constant(tau), self.args.dvae.hard)
        self.dvae_optimizer.apply_gradients(zip(gradients, self.dvae.trainable_weights))

        z_transformer_input, z_transformer_target = create_tokens(tf.stop_gradient(dvae_out['z_hard']))

        sm_out, sm_mets, gradients = slot_model.SlotModel.loss_and_grad(self.slot_model, z_transformer_input, z_transformer_target)
        # NOTE: if we put this inside tf.function then the performance becomes very bad
        self.main_optimizer.apply_gradients(zip(gradients, self.slot_model.trainable_weights))

        loss = dvae_mets['mse'] + sm_mets['cross_entropy']

        outputs = dict(
            dvae=dvae_out,
            slot_model=sm_out,
            iterates=dict(
                tau=tau,
                lr_warmup_factor=lr_warmup_factor)
            )
        metrics = dict(
            mse=dvae_mets['mse'],
            cross_entropy=sm_mets['cross_entropy'],
            loss=loss)

        self.step.assign_add(1)

        return loss, outputs, metrics

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
        return debug_args

    @staticmethod
    def defaults():
        default_args = SLATE.defaults()
        default_args.slot_model = slot_model.DynamicSlotModel.defaults()
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

        image = data['image']

        B, T, *_ = image.shape
        image = rearrange(image, 'b t h w c -> (b t) c h w')

        dvae_out, dvae_mets = self.dvae(image, tau, hard)

        z_transformer_input, z_transformer_target = create_tokens(tf.stop_gradient(dvae_out['z_hard']))

        #
        z_transformer_input = rearrange(z_transformer_input, '(b t) ... -> b t ...', b=B, t=T)
        z_transformer_target = rearrange(z_transformer_target, '(b t) ... -> b t ...', b=B, t=T)
        #

        sm_out, sm_mets = self.slot_model(z_transformer_input, z_transformer_target, data['action'], data['is_first'])

        # 
        sm_out['attns'] = rearrange(sm_out['attns'], 'b t ... -> (b t) ...')
        #

        return (
            dvae_out['recon'],
            sm_mets['cross_entropy'],
            dvae_mets['mse'],
            sm_out['attns'],
            dvae_out['z_hard']
        )


    def imagine(self, slots, actions):
        """
            slots: (b, k, ds)
            actions: (b, t, da)
        """
        bsize = slots.shape[0]
        imag_latent = self.slot_model.generate(slots, actions)
        z_gen = utils.bottle(self.slot_model.autoregressive_decode)(slots)
        recon_transformer = self.decode(z_gen)
        return recon_transformer

    def rollout(self, batch, seed_steps, pred_horizon):
        obs = batch['image'][:, :seed_steps + pred_horizon]
        act = batch['action'][:, :seed_steps + pred_horizon]
        is_first = batch['is_first'][:, :seed_steps + pred_horizon]
        recon_pred, recon_slots, recon_attns = self.call({'image': obs[:, :seed_steps], 'action': act[:, :seed_steps], 'is_first': is_first[:, :seed_steps]}, tau, hard)
        imag_pred, imag_slots, imag_attns = self.imagine(recon_slots[:, -1], act[:, seed_steps:])

        # rollout_ouptut
        # rollout_metrics


        # do reconstruct autoregressgive
        # then do imagine
        pass


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
        # global_step should be the same as self.step

        B, T, _ = data['action'].shape

        tau = cosine_anneal(
            step=self.step.numpy(),
            start_value=self.args.dvae.tau_start,
            final_value=self.args.dvae.tau_final,
            start_step=0,
            final_step=self.args.dvae.tau_steps)

        lr_warmup_factor = linear_warmup(
            step=self.step.numpy(),
            start_value=0.,
            final_value=1.0,
            start_step=0,
            final_step=self.args.slot_model.lr_warmup_steps)

        self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
        self.main_optimizer.lr = f32(self.args.lr_decay_factor * lr_warmup_factor * self.args.slot_model.lr)

        image = rearrange(data['image'], 'b t h w c -> (b t) c h w')
        dvae_out, dvae_mets, gradients = dvae.dVAE.loss_and_grad(self.dvae, image, tf.constant(tau), self.args.dvae.hard)
        self.dvae_optimizer.apply_gradients(zip(gradients, self.dvae.trainable_weights))

        z_transformer_input, z_transformer_target = create_tokens(tf.stop_gradient(dvae_out['z_hard']))

        sm_out, sm_mets, gradients = slot_model.DynamicSlotModel.loss_and_grad(self.slot_model, 
            rearrange(z_transformer_input, '(b t) ... -> b t ...', b=B, t=T), 
            rearrange(z_transformer_target, '(b t) ... -> b t ...', b=B, t=T),
            action=data['action'],
            is_first=data['is_first']
            # here, add is_first, action

            )
        # NOTE: if we put this inside tf.function then the performance becomes very bad
        self.main_optimizer.apply_gradients(zip(gradients, self.slot_model.trainable_weights))

        loss = dvae_mets['mse'] + sm_mets['cross_entropy']

        outputs = dict(
            dvae=dvae_out,
            slot_model=sm_out,
            iterates=dict(
                tau=tau,
                lr_warmup_factor=lr_warmup_factor)
            )
        metrics = dict(
            mse=dvae_mets['mse'],
            cross_entropy=sm_mets['cross_entropy'],
            loss=loss)

        self.step.assign_add(1)

        return loss, outputs, metrics
