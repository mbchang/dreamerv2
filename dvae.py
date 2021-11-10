from utils import *

from einops.layers.keras import Rearrange 
import tensorflow as tf
import tensorflow.keras.layers as tkl

class dVAE(tkl.Layer):
    
    def __init__(self, vocab_size, img_channels):
        super().__init__()
        
        self.encoder = tf.keras.Sequential([
            Conv2dBlock(img_channels, 64, 4, 4),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Rearrange('b c h w -> b h w c'),  # TODO: take this out if you reshape the entire input
            conv2d(64, vocab_size, 1),
            Rearrange('b h w c -> b c h w'),  # TODO: take this out if you reshape the entire input
        ])
        
        self.decoder = tf.keras.Sequential([
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),  # -> (B, 256, 14, 14)
            # nn.PixelShuffle(2),
            Rearrange('b c h w -> b h w c'),  # TODO: take this out if you reshape the entire input
            PixelShuffle(2),
            Rearrange('b h w c -> b c h w'),  # TODO: take this out if you reshape the entire input
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            # nn.PixelShuffle(2),
            Rearrange('b c h w -> b h w c'),  # TODO: take this out if you reshape the entire input
            PixelShuffle(2),
            Rearrange('b h w c -> b c h w'),  # TODO: take this out if you reshape the entire input
            Rearrange('b c h w -> b h w c'),  # TODO: take this out if you reshape the entire input
            conv2d(64, img_channels, 1),
            Rearrange('b h w c -> b c h w'),  # TODO: take this out if you reshape the entire input
        ])

class PixelShuffle(tkl.Layer):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def call(self, x):
        y = tf.nn.depth_to_space(input=x, block_size=self.upscale_factor, data_format='NHWC')
        return y