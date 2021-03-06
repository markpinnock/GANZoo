import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from .Blocks import (
    ProgGenFirstBlock,
    ProgGenLaterBlock,
    GANDiscBlock,
    StyleGenFirstBlock,
    StyleGenLaterBlock,
    MappingNet
    )

from utils.Losses import WeightClipConstraint


#-------------------------------------------------------------------------
""" Base class for both generator and discriminator """

class BaseGAN(keras.Model):

    def __init__(self, config, name):
        super().__init__(name=name)

        self.alpha = None
        self.blocks = []
        self.max_resolution = config["MAX_RES"]
        self.start_resolution = config["START_RES"]
        self.num_layers = int(np.log2(self.max_resolution)) - int(np.log2(self.start_resolution)) + 1
        self.resolutions = [self.start_resolution * 2 ** idx for idx in range(self.num_layers)]

    def call(self):
        raise NotImplementedError

#-------------------------------------------------------------------------
""" Discriminator class, inherits from BaseGAN """

class Discriminator(BaseGAN):

    """ Inputs:
            - d_nc: number of channels in first layer
            - GAN_type: implementation of GAN used
            - constraint_type: 'clip', 'maxnorm', or None
        Returns:
            - keras.Model """

    def __init__(self, config, name=None):
        super().__init__(config, name=name)

        self.channels = [np.min([(config["NDF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()

        self.blocks.append(GANDiscBlock(self.channels[0], None, config, name="block0"))

        for i in range(1, self.num_layers):
            new_block = GANDiscBlock(self.channels[i], self.blocks[i - 1], config, name=f"block{i}")
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self test on start up
        for i in range(self.num_layers):
            test = tf.zeros((2, 4 * (2 ** i), 4 * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=None).shape == (2, 1), self.blocks[i](test).shape
        
        for i in range(self.num_layers):
            test = tf.zeros((2, 4 * (2 ** i), 4 * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=0.5).shape == (2, 1), self.blocks[i](test, alpha=0.5).shape

    def call(self, x, scale, training=True):
        x = self.blocks[scale](x, self.alpha)
        
        return tf.squeeze(x)

#-------------------------------------------------------------------------
""" Generator class, inherits from BaseGAN """

class Generator(BaseGAN):

    """ Inputs:
            - config: configuration json
            - constraint type: """

    def __init__(self, config, name=None):
        super().__init__(config, name)

        latent_dims = config["LATENT_DIM"]
        self.channels = [np.min([(config["NGF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()

        if config["MODEL"] == "StyleGAN":
            self.StyleMap = MappingNet(config["STYLE_MAP_UNITS"], config["LATENT_DIM"], config["STYLE_MAP_LAYERS"], name="StyleMapping")
            _ = self.StyleMap(tf.zeros((2, latent_dims))) # Build implicitly until build method defined
            FirstBlock = StyleGenFirstBlock
            LaterBlock = StyleGenLaterBlock
        else:
            self.StyleMap = None
            FirstBlock = ProgGenFirstBlock
            LaterBlock = ProgGenLaterBlock

        self.blocks.append(FirstBlock(self.channels[0], self.resolutions[0], config, name="block0"))

        for i in range(1, self.num_layers):
            new_block = LaterBlock(self.channels[i], self.resolutions[i], self.blocks[i - 1], config, name=f"block{i}")
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self test on start up
        for i in range(0, self.num_layers):
            test = tf.zeros((2, latent_dims), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=None)[1].shape == (2, 4 * (2 ** i), 4 * (2 ** i), 3), self.blocks[i](test, alpha=None).shape

        for i in range(0, self.num_layers):
            test = tf.zeros((2, latent_dims), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=0.5)[1].shape == (2, 4 * (2 ** i), 4 * (2 ** i), 3), self.blocks[i](test, alpha=0.5).shape

    def call(self, z, scale, training=True):
        if self.StyleMap: z = self.StyleMap(z)
        _, rgb = self.blocks[scale](z, fade_alpha=self.alpha)

        return tf.nn.tanh(rgb)
