import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from abc import abstractclassmethod

from .Blocks import (
    ProGenFirstBlock,
    ProGenLaterBlock,
    GANDiscBlock,
    StyleGenFirstBlock,
    StyleGenLaterBlock,
    MappingNet
    )

from utils.Losses import WeightClipConstraint


#-------------------------------------------------------------------------
""" Base class for both generator and discriminator """

class BaseProStyleGANModel(keras.Model):

    def __init__(self, config, name):
        super().__init__(name=name)

        self.alpha = None
        self.blocks = []
        self.max_resolution = config["MAX_RES"]
        self.start_resolution = config["START_RES"]
        self.num_layers = int(np.log2(self.max_resolution)) - int(np.log2(self.start_resolution)) + 1
        self.resolutions = [self.start_resolution * 2 ** idx for idx in range(self.num_layers)]

    def build_generator_recursive(self, FirstBlock, LaterBlock, config):
        self.blocks.append(FirstBlock(self.channels[0], self.resolutions[0], config, name="block0"))

        for i in range(1, self.num_layers):
            new_block = LaterBlock(self.channels[i], self.resolutions[i], self.blocks[i - 1], config, name=f"block{i}")
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self-test
        for i in range(0, self.num_layers):
            test = tf.zeros((2, self.latent_dims), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=None)[1].shape == (2, self.start_resolution * (2 ** i), self.start_resolution * (2 ** i), 3), self.blocks[i](test, fade_alpha=None)[1].shape

        for i in range(0, self.num_layers):
            test = tf.zeros((2, self.latent_dims), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=0.5)[1].shape == (2, self.start_resolution * (2 ** i), self.start_resolution * (2 ** i), 3), self.blocks[i](test, fade_alpha=0.5)[1].shape

    @abstractclassmethod
    def call(self):
        raise NotImplementedError

#-------------------------------------------------------------------------
""" Discriminator class for ProGAN or StyleGAN, inherits from BaseGAN """

class ProStyleGANDiscriminator(BaseProStyleGANModel):

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
            test = tf.zeros((2, self.start_resolution * (2 ** i), self.start_resolution * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=None).shape == (2, 1), self.blocks[i](test).shape
        
        for i in range(self.num_layers):
            test = tf.zeros((2, self.start_resolution * (2 ** i), self.start_resolution * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=0.5).shape == (2, 1), self.blocks[i](test, fade_alpha=0.5).shape

    def call(self, x, scale, training=True):
        x = self.blocks[scale](x, self.alpha)
        
        return tf.squeeze(x)

#-------------------------------------------------------------------------
""" Generator class for ProGAN, inherits from BaseGAN """

class ProGANGenerator(BaseProStyleGANModel):

    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.latent_dims = config["LATENT_DIM"]
        self.output_activation = config["G_OUT"]
        assert self.output_activation in ["tanh", "linear"], "Choose tanh or linear output"
        self.channels = [np.min([(config["NGF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()

        self.build_generator_recursive(ProGenFirstBlock, ProGenLaterBlock, config)

    def call(self, z, scale, training=True):
        _, rgb = self.blocks[scale](z, fade_alpha=self.alpha)

        if self.output_activation == "tanh":
            return tf.nn.tanh(rgb)
        else:
            return rgb

#-------------------------------------------------------------------------
""" Generator class for StyleGAN, inherits from BaseGAN """

class StyleGANGenerator(BaseProStyleGANModel):

    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.latent_dims = config["LATENT_DIM"]
        self.output_activation = config["G_OUT"]
        assert self.output_activation in ["tanh", "linear"], "Choose tanh or linear output"
        self.channels = [np.min([(config["NGF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()

        self.StyleMap = MappingNet(config["STYLE_MAP_UNITS"], config["LATENT_DIM"], config["STYLE_MAP_LAYERS"], name="StyleMapping")
        _ = self.StyleMap(tf.zeros((2, self.latent_dims))) # Build implicitly until build method defined
        self.build_generator_recursive(StyleGenFirstBlock, StyleGenLaterBlock, config)

    def call(self, z, scale, training=True):
        w = self.StyleMap(z)
        _, rgb = self.blocks[scale](w, fade_alpha=self.alpha)

        if self.output_activation == "tanh":
            return tf.nn.tanh(rgb)
        else:
            return rgb
