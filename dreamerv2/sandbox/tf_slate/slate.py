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

    # @staticmethod
    # def defaults_debug():
    #     debug_args = SLATE.defaults()
    #     debug_args.log_interval = 8
    #     debug_args.batch_size = 5
    #     debug_args.vocab_size = 32
    #     debug_args.dvae = dvae.dVAE.defaults_debug()
    #     debug_args.slot_model = slot_model.SlotModel.defaults_debug()

    #     debug_args.mono_train = False

    #     debug_args.stop_gradient_input = True
    #     debug_args.stop_gradient_output = True

    #     debug_args.smooth_input = False

    #     return debug_args

    # testing smooth input
    @staticmethod
    def defaults_debug():
        debug_args = SLATE.defaults()
        debug_args.log_interval = 1
        debug_args.batch_size = 5
        debug_args.vocab_size = 32
        debug_args.dvae = dvae.dVAE.defaults_debug()
        debug_args.slot_model = slot_model.SlotModel.defaults_debug()

        debug_args.mono_train = True
        debug_args.smooth_input = False
        debug_args.stop_gradient_input = False
        debug_args.stop_gradient_output = True
        debug_args.nontokenized_embed = False
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

            stop_gradient_input=True,
            stop_gradient_output=True,
            nontokenized_embed=False,

            smooth_input=False
            ))
        return default_args

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_tokens = (args.image_size // 4) ** 2

        self.dvae = dvae.dVAE(
            vocab_size=args.vocab_size, 
            img_channels=args.img_channels, 
            d_model=args.slot_model.d_model, 
            sm_hard=args.dvae.sm_hard, 
            cnn_type=args.dvae.cnn_type,
            nontokenized_embed=args.nontokenized_embed)
        self.slot_model = slot_model.SlotModel(
            vocab_size=args.vocab_size, 
            num_tokens=self.num_tokens,
            nontokenized_embed=args.nontokenized_embed,
            args=args.slot_model)

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

        z_input, z_target = create_tokens(dvae_out['z_hard'])
        z_input, z_target = self.handle_stop_gradient(z_input, z_target)

        sm_loss, sm_out, sm_mets = self.slot_model(z_input, z_target, dvae_out['embeds'])

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
        z_logits, embeds = self.dvae.get_logits(image)
        z_hard = self.dvae.mode(z_logits)
        z_input, z_target = create_tokens(z_hard)
        return z_input, z_target, embeds

    # def image_to_sampled_tokens(self, image, tau, hard):
    #     z_logits, embeds = self.dvae.get_logits(image)
    #     z_sample = self.sample(z_logits, tau, hard)
    #     z_input, z_target = create_tokens(z_sample)
    #     return z_input, z_target

    def handle_stop_gradient(self, z_input, z_target):
        """
        assumes that z_hard is one_hot
        """
        if self.args.stop_gradient_input:
            z_input = tf.stop_gradient(z_input)
        if self.args.stop_gradient_output:
            z_target = tf.stop_gradient(z_target)
        return z_input, z_target

    @tf.function
    def reconstruct_autoregressive(self, image: tf.Tensor, eval: bool=False):
        """
        image: batch_size x img_channels x H x W
        """
        one_hot_tokens, _, embeds = self.image_to_argmax_tokens(image)
        emb_input = self.slot_model.embed_tokens(one_hot_tokens)
        slots, attns = self.slot_model.apply_slot_attn(emb_input[:, 1:])
        if self.slot_model.perceiver_output:
            pred = self.slot_model.perceiver_decode(slots)
            z_gen = self.slot_model.logits_to_tokens(pred)
        else:
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

        z_input, z_target = create_tokens(dvae_out['z_hard'])
        z_input, z_target = self.handle_stop_gradient(z_input, z_target)

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


    # def monolithic_train_step_smooth_input(self, image):
    #     iterates = self.get_iterates(self.step.numpy())

    #     self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
    #     self.main_optimizer.lr = f32(self.args.lr_decay_factor * iterates['lr_warmup_factor'] * self.args.slot_model.lr)

    #     with tf.GradientTape() as dvae_tape, tf.GradientTape() as sm_tape:

    #         dvae_loss, dvae_out, dvae_mets = self.dvae(image, iterates['tau'], self.args.dvae.hard)

    #         z_input, z_target = create_tokens(dvae_out['z_hard'])
    #         z_input, z_target = self.handle_stop_gradient(z_input, z_target)

    #         sm_loss, sm_out, sm_mets = self.slot_model(z_input, z_target, dvae_out['embeds'])

    #         outputs = dict(dvae=dvae_out, slot_model=sm_out)
    #         metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
    #         loss = dvae_loss + sm_loss


    #     dvae_grads = dvae_tape.gradient(loss, self.dvae.trainable_weights)
    #     sm_grads = sm_tape.gradient(loss, self.slot_model.trainable_weights)

    #     gradients = {'dvae': dvae_grads, 'slot_model': sm_grads}

    #     self.dvae_optimizer.apply_gradients(zip(gradients['dvae'], self.dvae.trainable_weights))
    #     self.main_optimizer.apply_gradients(zip(gradients['slot_model'], self.slot_model.trainable_weights))

    #     outputs = {'iterates': iterates, **outputs}

    #     self.step.assign_add(1)

    #     return loss, outputs, metrics



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
    """
    DynamicSLATE is like dreamer.wm
    DynamicSlotModel is like dreamer.rssm
    """

    @staticmethod
    def defaults_debug():
        debug_args = SLATE.defaults_debug()
        debug_args.slot_model = slot_model.DynamicSlotModel.defaults_debug()
        debug_args.vis_rollout = True
        debug_args.curr = True
        debug_args.curr_every = 1
        debug_args.e2e = False
        return debug_args

    @staticmethod
    def defaults():
        default_args = SLATE.defaults()
        default_args.slot_model = slot_model.DynamicSlotModel.defaults()
        default_args.vis_rollout = True
        default_args.curr = False
        default_args.curr_every = 20000
        default_args.e2e = False
        return default_args

    def __init__(self, seqlen, args, global_config):
        layers.Layer.__init__(self)
        self.args = args
        self.global_config = global_config  # a hack that we will remove once we integrate with RSSM

        self.dvae = dvae.dVAE(
            vocab_size=args.vocab_size, 
            img_channels=args.img_channels, 
            d_model=args.slot_model.d_model, 
            sm_hard=args.dvae.sm_hard, 
            cnn_type=args.dvae.cnn_type,
            nontokenized_embed=args.nontokenized_embed)
        self.dvae_optimizer = tf.keras.optimizers.Adam(args.dvae.lr, epsilon=1e-08)

        self.num_tokens = (args.image_size // 4) ** 2
        self.slot_model = slot_model.DynamicSlotModel(
            vocab_size=args.vocab_size, 
            num_tokens=self.num_tokens, 
            nontokenized_embed=args.nontokenized_embed,
            args=args.slot_model, 
            global_config=global_config)
        self.main_optimizer = tf.keras.optimizers.Adam(args.slot_model.lr, epsilon=1e-08)

        self.training = False
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.horizon_curriculum = Counter(
          initial_value=2 if args.curr else seqlen,
          final_value=seqlen,
          step_every=args.curr_every)

    def get_iterates(self, step):
        iterates = SLATE.get_iterates(self, step)

        lr_decay_factor = cosine_anneal(
            step=step,
            start_value=1.0,
            final_value=self.args.slot_model.min_lr_factor,
            start_step=self.args.slot_model.lr_warmup_steps,
            final_step=self.args.slot_model.lr_warmup_steps + self.args.slot_model.decay_steps)

        num_frames = self.horizon_curriculum.value(step)

        return {**iterates, **dict(lr_decay_factor=lr_decay_factor, num_frames=num_frames)}

    def call(self, data, tau, hard):
        """
        image: batch_size x img_channels x H x W
        """
        # assert False
        permute = lambda x: rearrange(x, '... h w c -> ... c h w')
        flatten = lambda x: rearrange(x, 'b t ... -> (b t) ...')
        unflatten = lambda x: rearrange(x, '(b t) ... -> b t ...', b=data['action'].shape[0])

        dvae_loss, dvae_out, dvae_mets = self.dvae(flatten(permute(data['image'])), tau, hard)

        z_input, z_target = create_tokens(dvae_out['z_hard'])
        z_input, z_target = self.handle_stop_gradient(z_input, z_target)

        sm_loss, sm_out, sm_mets = self.slot_model(
            unflatten(z_input), 
            unflatten(z_target), 
            data['action'], 
            data['is_first'],
            data['reward'],
            dvae_out['embeds'],
            )

        sm_out['attns'] = flatten(sm_out['attns'])

        outputs = dict(dvae=dvae_out, slot_model=sm_out)
        metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
        loss = dvae_loss + sm_loss
        return loss, outputs, metrics

    def rollout(self, batch, seed_steps, pred_horizon):
        batch_horizon = tf.nest.map_structure(lambda x: x[:, :seed_steps + pred_horizon], batch)
        batch_seed = tf.nest.map_structure(lambda x: x[:, :seed_steps], batch)

        permute = lambda x: rearrange(x, '... h w c -> ... c h w')
        flatten = lambda x: rearrange(x, 'b t ... -> (b t) ...')
        unflatten = lambda x: rearrange(x, '(b t) ... -> b t ...', b=batch['action'].shape[0])

        # image = flatten(permute(batch['image']))  # this should be batch_horizon
        image = flatten(permute(batch_horizon['image']))  # this should be batch_horizon
        z_input, z_target, embeds = map(unflatten, self.image_to_argmax_tokens(image))

        recon_output = self.slot_model.recon_autoregressive(z_input[:, :seed_steps], batch_seed['action'], batch_seed['is_first'], embeds)
        recon_ce = slot_model.SlotModel.cross_entropy_loss(recon_output['z_gen'], z_target[:, :seed_steps])
        if pred_horizon > 0:
            imag_output = self.slot_model.imag_autoregressive(
                # recon_output['slots'][:, -1], 
                # {'deter': recon_output['slots'][:, -1]}, 
                {'deter': recon_output['slots']['deter'][:, -1]}, 
                batch_horizon['action'][:, seed_steps:])
            imag_ce = slot_model.SlotModel.cross_entropy_loss(imag_output['z_gen'], z_target[:, seed_steps:])
            output = {'video': bottle(self.decode)(tf.concat((recon_output['z_gen'], imag_output['z_gen']), axis=1))}
            metrics = {'recon': recon_ce, 'imag': imag_ce}
        else:
            output = {'video': bottle(self.decode)(recon_output['z_gen'])}
            metrics = {'recon': recon_ce}
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
        data = tf.nest.map_structure(lambda x: x[:, :iterates['num_frames']], data)

        self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
        self.main_optimizer.lr = f32(iterates['lr_decay_factor'] * iterates['lr_warmup_factor'] * self.args.slot_model.lr)

        dvae_loss, dvae_out, dvae_mets, gradients = self.dvae.loss_and_grad(flatten(permute(data['image'])), tf.constant(iterates['tau']), self.args.dvae.hard)
        self.dvae_optimizer.apply_gradients(zip(gradients, self.dvae.trainable_weights))

        z_input, z_target = create_tokens(dvae_out['z_hard'])
        z_input, z_target = self.handle_stop_gradient(z_input, z_target)

        sm_loss, sm_out, sm_mets, gradients = self.slot_model.loss_and_grad(
            unflatten(z_input),
            unflatten(z_target),
            action=data['action'],
            is_first=data['is_first'],
            reward=data['reward']
            )
        # NOTE: if we put this inside tf.function then the performance becomes very bad
        self.main_optimizer.apply_gradients(zip(gradients, self.slot_model.trainable_weights))

        loss = dvae_loss + sm_loss

        outputs = dict(dvae=dvae_out, slot_model=sm_out, iterates=iterates)
        metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
        self.step.assign_add(1)

        return loss, outputs, metrics


    def monolithic_train_step(self, data):
        permute = lambda x: rearrange(x, '... h w c -> ... c h w')
        flatten = lambda x: rearrange(x, 'b t ... -> (b t) ...')
        unflatten = lambda x: rearrange(x, '(b t) ... -> b t ...', b=data['action'].shape[0])

        iterates = self.get_iterates(self.step.numpy())
        data = tf.nest.map_structure(lambda x: x[:, :iterates['num_frames']], data)

        self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
        self.main_optimizer.lr = f32(iterates['lr_decay_factor'] * iterates['lr_warmup_factor'] * self.args.slot_model.lr)

        with tf.GradientTape() as dvae_tape, tf.GradientTape() as sm_tape:

            dvae_loss, dvae_out, dvae_mets = self.dvae(flatten(permute(data['image'])), tf.constant(iterates['tau']), self.args.dvae.hard)

            z_input, z_target = create_tokens(dvae_out['z_hard'])
            z_input, z_target = self.handle_stop_gradient(z_input, z_target)

            sm_loss, sm_out, sm_mets = self.slot_model(
                unflatten(z_input),
                unflatten(z_target),
                action=data['action'],
                is_first=data['is_first'],
                reward=data['reward'],
                embeds=dvae_out['embeds'],
                )

            outputs = dict(dvae=dvae_out, slot_model=sm_out)
            metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
            loss = dvae_loss + sm_loss

        dvae_grads = dvae_tape.gradient(loss, self.dvae.trainable_weights)
        sm_grads = sm_tape.gradient(loss, self.slot_model.trainable_weights)

        gradients = {'dvae': dvae_grads, 'slot_model': sm_grads}

        self.dvae_optimizer.apply_gradients(zip(gradients['dvae'], self.dvae.trainable_weights))
        self.main_optimizer.apply_gradients(zip(gradients['slot_model'], self.slot_model.trainable_weights))

        outputs = {'iterates': iterates, **outputs}

        # ################################################

        self.step.assign_add(1)

        return loss, outputs, metrics

    # this will probably look a lot like rollout
    def e2e_monolithic_train_step(self, data):
        permute = lambda x: rearrange(x, '... h w c -> ... c h w')
        flatten = lambda x: rearrange(x, 'b t ... -> (b t) ...')
        unflatten = lambda x: rearrange(x, '(b t) ... -> b t ...', b=data['action'].shape[0])

        iterates = self.get_iterates(self.step.numpy())
        data = tf.nest.map_structure(lambda x: x[:, :iterates['num_frames']], data)

        self.dvae_optimizer.lr = f32(self.args.lr_decay_factor * self.args.dvae.lr)
        self.main_optimizer.lr = f32(iterates['lr_decay_factor'] * iterates['lr_warmup_factor'] * self.args.slot_model.lr)

        with tf.GradientTape() as dvae_tape, tf.GradientTape() as sm_tape:
            
            dvae_loss, dvae_out, dvae_mets = self.dvae(flatten(permute(data['image'])), tf.constant(iterates['tau']), self.args.dvae.hard)

            z_input, z_target = create_tokens(dvae_out['z_hard'])
            z_input, z_target = self.handle_stop_gradient(z_input, z_target)

            sm_loss, sm_out, sm_mets = self.slot_model(
                unflatten(z_input),
                unflatten(z_target),
                action=data['action'],
                is_first=data['is_first'],
                reward=data['reward'],
                embeds=dvae_out['embeds'],
                )

            ################################################
            # sm_out['pred']: (B, N, H*W, V). the thing that gets fed into log_softmax for cross entropy. So we can also feed it into softmax for cross entropy.
            sm_sample = self.dvae.sample(sm_out['pred'], tf.constant(iterates['tau']), self.args.dvae.hard, dim=-1)
            recon = flatten(bottle(self.decode)(sm_sample))
            concat = lambda x: rearrange(x, 'n b ... -> (n b) ...')
            image = flatten(permute(concat([data['image'], data['image']])))
            mse = tf.math.reduce_sum((image - recon) ** 2) / image.shape[0]

            outputs = dict(dvae=dvae_out, slot_model=sm_out)
            metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
            loss = tf.reduce_mean([dvae_mets['mse'], mse]) + sm_loss

            # ************
            # outputs = dict(dvae=dvae_out, slot_model=sm_out)
            # metrics = dict(dvae=dvae_mets, slot_model=sm_mets)
            # loss = dvae_loss + sm_loss
            ################################################

        dvae_grads = dvae_tape.gradient(loss, self.dvae.trainable_weights)
        sm_grads = sm_tape.gradient(loss, self.slot_model.trainable_weights)

        gradients = {'dvae': dvae_grads, 'slot_model': sm_grads}

        self.dvae_optimizer.apply_gradients(zip(gradients['dvae'], self.dvae.trainable_weights))
        self.main_optimizer.apply_gradients(zip(gradients['slot_model'], self.slot_model.trainable_weights))

        outputs = {'iterates': iterates, **outputs}

        # ################################################

        self.step.assign_add(1)

        return loss, outputs, metrics


