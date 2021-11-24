from utils import *

from einops.layers.keras import Rearrange
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as tkl

class dVAE(tkl.Layer):

    @staticmethod
    def defaults_debug():
        debug_args = dVAE.defaults()
        debug_args.tau_steps=3
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            lr=3e-4,

            tau_start=1.0,
            tau_final=0.1,
            tau_steps=30000,

            hard=False,
            ))
        return default_args
    
    def __init__(self, vocab_size, img_channels):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.encoder = tf.keras.Sequential([
            Rearrange('b c h w -> b h w c'),
            Conv2dBlock(img_channels, 64, 4, 4),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            conv2d(64, vocab_size, 1),
            Rearrange('b h w c -> b c h w'),
        ])
        
        self.decoder = tf.keras.Sequential([
            Rearrange('b c h w -> b h w c'),
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            PixelShuffle(2),
            conv2d(64, img_channels, 1),
            Rearrange('b h w c -> b c h w'),
        ])

    def get_logits(self, image):
        z_logits = tf.nn.log_softmax(self.encoder(image), axis=1)
        return z_logits

    def sample(self, z_logits, tau, hard):
        z = gumbel_softmax(z_logits, tau, hard, dim=1)
        return z

    def mode(self, z_logits):
        z_hard = tf.math.argmax(z_logits, axis=1)
        z_hard = tf.cast(rearrange(tf.one_hot(z_hard, depth=self.vocab_size), 'b h w v -> b v h w'), tf.float32)
        return z_hard

    def call(self, image, tau, hard):
        # dvae encode
        z_logits = self.get_logits(image)
        z = self.sample(z_logits, tau, hard)
        # dvae recon
        recon = self.decoder(z)
        mse = tf.math.reduce_sum((image - recon) ** 2) / image.shape[0]
        # hard z
        z_hard = self.sample(z_logits, tau, True)
        # ship
        outputs = {'recon': recon,'z_hard': z_hard}
        metrics = {'mse': mse, 'dvae/loss': mse}
        return outputs, metrics

    # @staticmethod
    @tf.function
    def loss_and_grad(self, image, tau, hard):
        with tf.GradientTape() as tape:
            outputs, metrics = self(image, tau, hard)
        gradients = tape.gradient(metrics['dvae/loss'], self.trainable_weights)
        return outputs, metrics, gradients


class PixelShuffle(tkl.Layer):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def call(self, x):
        y = tf.nn.depth_to_space(input=x, block_size=self.upscale_factor, data_format='NHWC')
        return y