from utils import *

from einops.layers.keras import Rearrange 
import tensorflow as tf
import tensorflow.keras.layers as tkl

class dVAE(tkl.Layer):
    
    def __init__(self, vocab_size, img_channels):
        super().__init__()
        
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
            # nn.PixelShuffle(2),
            PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            # nn.PixelShuffle(2),
            PixelShuffle(2),
            conv2d(64, img_channels, 1),
            Rearrange('b h w c -> b c h w'),
        ])

    def call(self, image, tau, hard):
        import time
        t0 = time.time()
        B, C, H, W = image.shape

        # dvae encode
        # z_logits = F.log_softmax(self.encoder(image), dim=1)
        z_logits = tf.nn.log_softmax(self.encoder(image), axis=1)
        _, _, H_enc, W_enc = z_logits.shape
        z = gumbel_softmax(z_logits, tau, hard, dim=1)

        # dvae recon
        recon = self.decoder(z)
        # mse = ((image - recon) ** 2).sum() / B
        mse = tf.math.reduce_sum((image - recon) ** 2) / B

        # hard z
        # z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()
        z_hard = tf.stop_gradient(gumbel_softmax(z_logits, tau, True, dim=1))

        print(f'That took {time.time() - t0} seconds.')
        return recon, z_hard, mse


class PixelShuffle(tkl.Layer):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def call(self, x):
        y = tf.nn.depth_to_space(input=x, block_size=self.upscale_factor, data_format='NHWC')
        return y