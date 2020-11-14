import tensorflow as tf
import tensorflow.keras as keras

from networks.Layers import MinibatchStd, PixelNorm, FadeInLayer
from utils.TrainFuncs import WeightClipConstraint


class GANBlock(keras.layers.Layer):
    def __init__(self, nc, initialiser, weight_const, batchnorm, transpose):
        super(GANBlock, self).__init__()
        self.batchnorm = batchnorm
        # TODO: change BN to LN in WGAN-GP
        if transpose:
            self.conv = keras.layers.Conv2DTranspose(nc, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=initialiser, kernel_constraint=weight_const)
        else:
            self.conv = keras.layers.Conv2D(nc, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=initialiser, kernel_constraint=weight_const)
        
        if self.batchnorm:
            self.bn = keras.layers.BatchNormalization()
        
        def call(self, x):
            x = self.conv(x)

            if self.batchnorm:
                x = self.bn(x)
            
            return tf.nn.leaky_relu(x, alpha=0.2)


class ProgGANDiscBlock(keras.layers.Layer):
    def __init__(self, prev_block, initialiser, weight_const):
        super(ProgGANDiscBlock, self).__init__()
        
        self.conv1 = keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
        self.conv2 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
        self.conv3 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)

        if prev_block == None:
            self.out = keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding="VALID", kernel_initializer=initialiser, kernel_constraint=weight_const)
        else:
            self.down = keras.layers.AveragePooling2D()
        
        self.mb_stats = MinibatchStd()
        self.prev_block = prev_block
        self.fade_in = FadeInLayer()

    def call(self, x, alpha, first_block=True):
        if first_block:
            dn = self.down(x)
            x = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)
            dn = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)

        if self.prev_block == None:
            x = self.mb_stats(x)

        x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)
        x = tf.nn.leaky_relu(self.conv3(x), alpha=0.2)

        if self.prev_block == None:
            x = self.out(x)
        else:
            x = self.down(x)
            x = self.fade_in([dn, x], alpha)
            x = self.prev_block(x, first_block=False)

        return x


class ProgGANGenBlock(keras.layers.Layer):
    def __init__(self, latent_dims, prev_block, initialiser, weight_const):
        super(ProgGANGenBlock, self).__init__()

        self.pixel_norm = PixelNorm()
        self.fade_in = FadeInLayer()
        
        # If first block
        if prev_block == None:
            self.reshaped = keras.layers.Reshape((1, 1, latent_dims))
            self.conv1 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding="VALID", kernel_initializer=initialiser, kernel_constraint=weight_const)
        else:
            self.up = keras.layers.UpSampling2D()
        
        self.prev_block = prev_block
        self.conv2 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
        self.conv3 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)

        self.out = keras.layers.Conv2D(3, (1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)

    def call(self, x, alpha, last_block=True):

        if self.prev_block == None:
            x = self.reshaped(x)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv1(x)), alpha=0.2)
        else:
            x = self.prev_block(x, last_block=False)
            up = self.up(x)
        
        x = tf.nn.leaky_relu(self.pixel_norm(self.conv2(up)), alpha=0.2)
        x = tf.nn.leaky_relu(self.pixel_norm(self.conv3(x)), alpha=0.2)

        if last_block:
            up = self.out(up)
            x = self.out(x)
            x = self.fade_in(alpha, [up, x])
        
        return x


class Discriminator(keras.Model):

    """ Discriminator model for GAN

        Inputs:
            - d_nc: number of channels in first layer
            - initaliser: e.g. keras.initalizers.RandomNormal()
            - constraint_type: 'clip', 'maxnorm', or None

        Returns keras.Model """

    def __init__(self, d_nc, initialiser, constraint_type):
        super(Discriminator, self).__init__()

        if constraint_type == "clip":
            weight_const = WeightClipConstraint(0.01)
        elif constraint_type == "maxnorm":
            weight_const = keras.constraints.MaxNorm(1)
        else:
            weight_const = None

        self.blocks = []
        self.blocks.append(ProgGANDiscBlock(None, initialiser, weight_const))

        for i in range(4):
            self.blocks.append(ProgGANDiscBlock(self.blocks[i], initialiser, weight_const))

        # Recursive self test on start up
        for i in range(4):
            test = tf.zeros((2, 4 * (2 ** i), 4 * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test).shape == (2, 1, 1, 1), self.blocks[i](test).shape

    def call(self, x, scale, training=True):
        x = self.blocks[scale](x)

        return tf.squeeze(x)

# TODO: subclass Generator and Discriminator
class Generator(keras.Model):

    """ Generator model for GAN
        - latient_dims: size of latent distribution
        - g_nc: number of channels in first layer
        - initaliser: e.g. keras.initalizers.RandomNormal() """

    def __init__(self, latent_dims, g_nc, initialiser, constraint_type):
        super(Generator, self).__init__()

        if constraint_type == "maxnorm":
            weight_const = keras.constraints.MaxNorm(1)
        else:
            weight_const = None

        self.blocks = []
        self.blocks.append(ProgGANGenBlock(latent_dims, None, initialiser, weight_const))

        for i in range(4):
            self.blocks.append(ProgGANGenBlock(latent_dims, self.blocks[i], initialiser, weight_const))

        # Recursive self test on start up
        for i in range(4):
            test = tf.zeros((2, 128), dtype=tf.float32)
            assert self.blocks[i](test).shape == (2, 4 * (2 ** i), 4 * (2 ** i), 3), self.blocks[i](test).shape

    def call(self, x, scale, training=True):
        x = self.blocks[scale](x)

        return tf.nn.tanh(x)

