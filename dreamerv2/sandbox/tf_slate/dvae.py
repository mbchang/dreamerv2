from utils import *

import einops as eo
from einops.layers.keras import Rearrange
import ml_collections
import tensorflow as tf
import tensorflow.keras.layers as tkl

class PaddedConv2D(tkl.Layer):
    def __init__(self, padding=0, **kwargs):
        super().__init__()
        self.m = tf.keras.Sequential([
            tkl.ZeroPadding2D(padding=padding),
            tkl.Conv2D(**kwargs)
            ])

    def call(self, x):
        return self.m(x)

class PaddedConv2DTranspose(tkl.Layer):
    def __init__(self, padding=0, dilation_rate=1, **kwargs):
        super().__init__()
        self.m = tf.keras.Sequential([
            tkl.Conv2DTranspose(**kwargs),
            tkl.Cropping2D(cropping=padding),
            ])

    def call(self, x):
        return self.m(x)


class ResBlock(tkl.Layer):
    def __init__(self, chan):
        super().__init__()
        self.net = tf.keras.Sequential([
            PaddedConv2D(filters=chan, kernel_size=3, padding=1),
            tkl.ReLU(),
            PaddedConv2D(filters=chan, kernel_size=3, padding=1),
            tkl.ReLU(),
            PaddedConv2D(filters=chan, kernel_size=1),
        ])

    def call(self, x):
        return self.net(x) + x


class dVAEWeakEncoder(tkl.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = tf.keras.Sequential([
                Conv2dBlock(in_channels, 64, 4, 4),
                Conv2dBlock(64, 64, 1, 1),
                Conv2dBlock(64, 64, 1, 1),
                Conv2dBlock(64, 64, 1, 1),
                Conv2dBlock(64, 64, 1, 1),
                Conv2dBlock(64, 64, 1, 1),
                Conv2dBlock(64, 64, 1, 1),
            ])

    def call(self, image):
        return self.net(image)


class dVAEShallowWeakEncoder(tkl.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = tf.keras.Sequential([
                Conv2dBlock(in_channels, 64, 4, 4),
                Conv2dBlock(64, 64, 1, 1),
                Conv2dBlock(64, 64, 1, 1),
            ])
    
    def call(self, image):
        return self.net(image)


class dVAEStrongEncoder(tkl.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NOTE: you can probably get away with 32 filters instead
        conv = lambda **kwargs: tf.keras.Sequential([PaddedConv2D(**kwargs), tkl.ReLU()])
        self.net = tf.keras.Sequential([
            conv(filters=64, kernel_size=4, strides=2, padding=1),
            conv(filters=64, kernel_size=4, strides=2, padding=1),
            ResBlock(chan=64),
            ])
    
    def call(self, image):
        return self.net(image)


class GenericEncoder(tkl.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NOTE: you can probably get away with 32 filters instead
        conv = lambda **kwargs: tf.keras.Sequential([PaddedConv2D(**kwargs), tkl.LayerNormalization(),tkl.ReLU()])
        self.net = tf.keras.Sequential([
            conv(filters=32, kernel_size=4, strides=2, padding=1),
            conv(filters=64, kernel_size=4, strides=2, padding=1),
            ])
    
    def call(self, image):
        return self.net(image)


class dVAEWeakDecoder(tkl.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = tf.keras.Sequential([
                Conv2dBlock(in_channels, 64, 1),
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
                conv2d(64, out_channels, 1),
            ])

    def call(self, image):
        return self.net(image)


class dVAEShallowWeakDecoder(tkl.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = tf.keras.Sequential([
                Conv2dBlock(in_channels, 64, 1),
                Conv2dBlock(64, 64, 3, 1, 1),
                Conv2dBlock(64, 64 * 2 * 2, 1),
                PixelShuffle(2),
                Conv2dBlock(64, 64, 3, 1, 1),
                Conv2dBlock(64, 64 * 2 * 2, 1),
                PixelShuffle(2),
                conv2d(64, out_channels, 1),
            ])

    def call(self, image):
        return self.net(image)


class dVAEStrongDecoder(tkl.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        convT = lambda **kwargs: tf.keras.Sequential([PaddedConv2DTranspose(**kwargs), tkl.ReLU()])
        self.net = tf.keras.Sequential([
            PaddedConv2D(filters=64, kernel_size=1),
            ResBlock(chan=64),
            convT(filters=64, kernel_size=4, strides=2, padding=1),
            convT(filters=64, kernel_size=4, strides=2, padding=1),
            PaddedConv2D(filters=out_channels, kernel_size=1),
            ])

    def call(self, image):
        return self.net(image)

class GenericDecoder(tkl.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = lambda **kwargs: tf.keras.Sequential([PaddedConv2D(**kwargs), tkl.LayerNormalization(),tkl.ReLU()])
        convT = lambda **kwargs: tf.keras.Sequential([PaddedConv2DTranspose(**kwargs), tkl.LayerNormalization(), tkl.ReLU()])
        self.net = tf.keras.Sequential([
            conv(filters=64, kernel_size=1),
            convT(filters=32, kernel_size=4, strides=2, padding=1),
            convT(filters=16, kernel_size=4, strides=2, padding=1),
            PaddedConv2D(filters=out_channels, kernel_size=1),
            ])

    def call(self, image):
        return self.net(image)

class dVAE(tkl.Layer):

    # @staticmethod
    # def defaults_debug():
    #     debug_args = dVAE.defaults()
    #     debug_args.tau_steps=3
    #     debug_args.sm_hard=True
    #     debug_args.cnn_type='sweak'
    #     return debug_args

    # testing smooth input
    @staticmethod
    def defaults_debug():
        debug_args = dVAE.defaults()
        debug_args.tau_steps=3
        debug_args.sm_hard=True
        debug_args.cnn_type='generic'
        return debug_args

    @staticmethod
    def defaults():
        default_args = ml_collections.ConfigDict(dict(
            lr=3e-4,

            tau_start=1.0,
            tau_final=0.1,
            tau_steps=30000,

            hard=False,
            sm_hard=True,
            cnn_type='sweak'
            ))
        return default_args
        
    def __init__(self, vocab_size, img_channels, sm_hard, cnn_type):
        super().__init__()
        self.vocab_size = vocab_size
        self.sm_hard = sm_hard
        
        if cnn_type == 'weak':
            self.encoder = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                dVAEWeakEncoder(img_channels, vocab_size),
                Rearrange('b h w c -> b c h w'),
            ])

            self.token_head = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                conv2d(64, vocab_size, 1),
                Rearrange('b h w c -> b c h w'),
            ])
            
            self.decoder = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                dVAEWeakDecoder(vocab_size, img_channels),
                Rearrange('b h w c -> b c h w'),
            ])
        elif cnn_type == 'sweak':
            self.encoder = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                dVAEShallowWeakEncoder(img_channels, vocab_size),
                Rearrange('b h w c -> b c h w'),
            ])

            self.token_head = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                conv2d(64, vocab_size, 1),
                Rearrange('b h w c -> b c h w'),
            ])
            
            self.decoder = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                dVAEShallowWeakDecoder(vocab_size, img_channels),
                Rearrange('b h w c -> b c h w'),
            ])  
        elif cnn_type == 'strong':
            self.encoder = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                dVAEStrongEncoder(img_channels, vocab_size),
                Rearrange('b h w c -> b c h w'),
            ])
            self.token_head = conv2d(64, vocab_size, 1)

            self.token_head = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                conv2d(64, vocab_size, 1),
                Rearrange('b h w c -> b c h w'),
            ])
            
            self.decoder = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                dVAEStrongDecoder(vocab_size, img_channels),
                Rearrange('b h w c -> b c h w'),
            ])
        elif cnn_type == 'generic':
            self.encoder = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                GenericEncoder(img_channels, vocab_size),
                Rearrange('b h w c -> b c h w'),
            ])

            self.token_head = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                conv2d(64, vocab_size, 1),
                Rearrange('b h w c -> b c h w'),
            ])

            self.decoder = tf.keras.Sequential([
                Rearrange('b c h w -> b h w c'),
                GenericDecoder(vocab_size, img_channels),
                Rearrange('b h w c -> b c h w'),
            ])
        else:
            raise NotImplementedError

        # self.token_head = conv2d(64, out_channels, 1)

    def get_logits(self, image):
        # z_logits = tf.nn.log_softmax(self.encoder(image), axis=1)
        z_logits = tf.nn.log_softmax(self.token_head(self.encoder(image)), axis=1)  # TODO
        # z_logits = tf.nn.log_softmax(self.token_head(self.encoder(image)), axis=-1)  # TODO
        return z_logits

    def sample(self, z_logits, tau, hard, dim=1):
        z = gumbel_softmax(z_logits, tau, hard, dim)
        return z

    def mode(self, z_logits):
        z_hard = tf.math.argmax(z_logits, axis=1)
        z_hard = tf.cast(rearrange(tf.one_hot(z_hard, depth=self.vocab_size), 'b h w v -> b v h w'), tf.float32)
        return z_hard

    def sample_encode(self, image, tau, hard):
        z_logits = self.get_logits(image)
        z_hard = self.sample(z_logits, tau, hard)
        return z_hard

    def mode_encode(self, image):
        z_logits = self.get_logits(image)
        z_hard = self.mode(z_logits)
        return z_hard

    def call(self, image, tau, hard):
        # dvae encode
        z_logits = self.get_logits(image)  # (B, V, H, W)
        z = self.sample(z_logits, tau, hard)
        # dvae recon
        recon = self.decoder(z)
        mse = tf.math.reduce_sum((image - recon) ** 2) / image.shape[0]
        # hard z
        # z_hard = self.sample(z_logits, tau, True)
        z_hard = self.sample(z_logits, tau, self.sm_hard)
        # ship
        outputs = {'recon': recon,'z_hard': z_hard}
        metrics = {'mse': mse, 'dvae/loss': mse}
        loss = mse
        return loss, outputs, metrics

    @tf.function
    def loss_and_grad(self, image, tau, hard):
        with tf.GradientTape() as tape:
            loss, outputs, metrics = self(image, tau, hard)
        gradients = tape.gradient(loss, self.trainable_weights)
        return loss, outputs, metrics, gradients


class PixelShuffle(tkl.Layer):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def call(self, x):
        y = tf.nn.depth_to_space(input=x, block_size=self.upscale_factor, data_format='NHWC')
        return y