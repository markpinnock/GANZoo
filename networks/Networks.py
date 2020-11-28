import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from networks.Layers import ProgGANGenBlock, ProgGANDiscBlock
from utils.TrainFuncs import WeightClipConstraint


#-------------------------------------------------------------------------
""" Base class for both generator and discriminator """

class BaseGAN(keras.Model):

    def __init__(self, config, constraint_type):
        super(BaseGAN, self).__init__()
        
        if constraint_type == "clip":
            self.weight_const = WeightClipConstraint(0.01)
        elif constraint_type == "maxnorm":
            # self.weight_const = keras.constraints.MaxNorm(1)
            self.weight_const = None
        else:
            self.weight_const = None

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

    def __init__(self, config, constraint_type):
        super(Discriminator, self).__init__(config, constraint_type)

        self.channels = [np.min([(config["NDF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()

        self.blocks.append(ProgGANDiscBlock(self.channels[0], self.resolutions[0], None, config, self.weight_const))

        for i in range(1, self.num_layers):
            new_block = ProgGANDiscBlock(self.channels[i], self.resolutions[i], self.blocks[i - 1], config, self.weight_const)
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self test on start up
        for i in range(self.num_layers):
            test = tf.zeros((2, 4 * (2 ** i), 4 * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, alpha=None).shape == (2, 1), self.blocks[i](test).shape
        
        for i in range(self.num_layers):
            test = tf.zeros((2, 4 * (2 ** i), 4 * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, alpha=0.5).shape == (2, 1), self.blocks[i](test, alpha=0.5).shape

    def call(self, x, scale, training=True):
        x = self.blocks[scale](x, self.alpha)
        
        return tf.squeeze(x)

#-------------------------------------------------------------------------
""" Generator class, inherits from BaseGAN """

class Generator(BaseGAN):

    """ Inputs:
            - config: configuration json
            - constraint type: """

    def __init__(self, config, constraint_type):
        super(Generator, self).__init__(config, constraint_type)

        latent_dims = config["LATENT_DIM"]
        self.channels = [np.min([(config["NGF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()

        self.blocks.append(ProgGANGenBlock(self.channels[0], self.resolutions[0], None, config, self.weight_const))

        for i in range(1, self.num_layers):
            new_block = ProgGANGenBlock(self.channels[i], self.resolutions[i], self.blocks[i - 1], config, self.weight_const)
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self test on start up
        for i in range(0, self.num_layers):
            test = tf.zeros((2, latent_dims), dtype=tf.float32)
            assert self.blocks[i](test, alpha=None).shape == (2, 4 * (2 ** i), 4 * (2 ** i), 3), self.blocks[i](test, alpha=None).shape

        for i in range(0, self.num_layers):
            test = tf.zeros((2, latent_dims), dtype=tf.float32)
            assert self.blocks[i](test, alpha=0.5).shape == (2, 4 * (2 ** i), 4 * (2 ** i), 3), self.blocks[i](test, alpha=0.5).shape

    def call(self, x, scale, training=True):
        x = self.blocks[scale](x, self.alpha)

        return tf.nn.tanh(x)

