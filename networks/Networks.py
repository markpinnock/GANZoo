import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from networks.Layers import ProgGANGenBlock, ProgGANDiscBlock
from utils.TrainFuncs import WeightClipConstraint


class BaseGAN(keras.Model):
    
    """ Base class for generator and discriminator """

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
        self.resolution = config["MAX_RES"]
        self.num_layers = int(np.log2(self.resolution)) - 1

    def call(self):
        raise NotImplementedError


class Discriminator(BaseGAN):

    """ Discriminator model for GAN

        Inputs:
            - d_nc: number of channels in first layer
            - GAN_type: implementation of GAN used
            - constraint_type: 'clip', 'maxnorm', or None

        Returns keras.Model """

    def __init__(self, config, constraint_type):
        super(Discriminator, self).__init__(config, constraint_type)

        self.channels = [np.min([(config["NDF"] * 2 ** i), self.resolution]) for i in range(self.num_layers) ]
        self.channels.reverse()
        
        self.blocks.append(ProgGANDiscBlock(self.channels[0], None, config["MAX_RES"], config["MODEL"], self.weight_const))

        for i in range(1, self.num_layers):
            new_block = ProgGANDiscBlock(self.channels[i], self.blocks[i - 1], config["MAX_RES"], config["MODEL"], self.weight_const)
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


class Generator(BaseGAN):

    """ Generator model for GAN
        - latient_dims: size of latent distribution
        - g_nc: number of channels in first layer
        - initaliser: e.g. keras.initalizers.RandomNormal() """

    def __init__(self, config, constraint_type):
        super(Generator, self).__init__(config, constraint_type)

        latent_dims = np.min([config["LATENT_DIM"], 512])
        self.channels = [np.min([(config["NGF"] * 2 ** i), self.resolution]) for i in range(self.num_layers) ]
        self.channels.reverse()

        self.blocks.append(ProgGANGenBlock(latent_dims, self.channels[0], None, config["MODEL"], self.weight_const))

        for i in range(1, self.num_layers):
            new_block = ProgGANGenBlock(latent_dims, self.channels[i], self.blocks[i - 1], config["MODEL"], self.weight_const)
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

