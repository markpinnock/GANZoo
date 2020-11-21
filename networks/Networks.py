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
    def __init__(self, next_block, initialiser, weight_const):
        super(ProgGANDiscBlock, self).__init__()

        # conv1 only used if first block
        self.next_block = next_block
        self.from_rgb = keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
        
        # If this is last discriminator block, collapse to prediction
        if next_block == None:
            self.mb_stats = MinibatchStd()
            self.conv2 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.out = keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding="VALID", kernel_initializer=initialiser, kernel_constraint=weight_const)
        
        # If next blocks exist, conv and downsample
        else:
            self.conv2 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.conv3 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.downsample = keras.layers.AveragePooling2D()
            self.fade_in = FadeInLayer()

    def call(self, x, alpha=None, first_block=True):
        
        # If fade in, cache downsampled input image
        if first_block and alpha != None and self.next_block != None:
            next_rgb = self.downsample(x)
            # Set from_rgb weights in next block to untrainable to avoid missing gradients
            self.next_block.from_rgb.trainable = False
            next_rgb = tf.nn.leaky_relu(self.next_block.from_rgb(next_rgb), alpha=0.2)

        # If the very first block, perform 1x1 conv
        if first_block:
            x = tf.nn.leaky_relu(self.from_rgb(x), alpha=0.2)

        # If this is not the last block
        if self.next_block != None:
            x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)
            x = tf.nn.leaky_relu(self.conv3(x), alpha=0.2)
            x = self.downsample(x)

            # If fade in, merge with cached layer
            if first_block and alpha != None and self.next_block != None:
                x = self.fade_in(alpha, [next_rgb, x])
            
            x = self.next_block(x, alpha=None, first_block=False)
        
        # If this is the last block
        else:
            x = self.mb_stats(x)
            x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)
            x = self.out(x)

        return x


class ProgGANGenBlock(keras.layers.Layer):
    def __init__(self, latent_dims, prev_block, initialiser, weight_const):
        super(ProgGANGenBlock, self).__init__()

        self.prev_block = prev_block
        self.pixel_norm = PixelNorm()
        
        # If this is first generator block, reshape latent noise
        if prev_block == None:
            self.reshaped = keras.layers.Reshape((1, 1, latent_dims))
            self.conv1 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding="VALID", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.conv2 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
        
        # If previous blocks exist, we use those
        else:
            self.upsample = keras.layers.UpSampling2D()
            self.conv1 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.conv2 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.fade_in = FadeInLayer()
        
        # Output to rgb
        self.to_rgb = keras.layers.Conv2D(3, (1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)

    def call(self, x, alpha=None, last_block=True):

        # If first block, upsample noise
        if self.prev_block == None:
            x = self.reshaped(x)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv1(x)), alpha=0.2)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv2(x)), alpha=0.2)
        
        # If previous blocks, upsample to_rgb and cache for fade in
        else:
            prev_x, prev_rgb = self.prev_block(x, alpha=None, last_block=False)
            # Set to_rgb weights in prev block to untrainable to avoid missing gradients
            self.prev_block.to_rgb.trainable = False
            prev_x = self.upsample(prev_x)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv1(prev_x)), alpha=0.2)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv2(x)), alpha=0.2)

        # Create output image
        rgb = self.to_rgb(x)

        # If fade in, merge cached prev block and this block
        if alpha != None and self.prev_block != None:
            prev_rgb = self.upsample(prev_rgb)
            rgb = self.fade_in(alpha, [prev_rgb, rgb])
        
        if last_block:
            return rgb
        else:
            return x, rgb


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

        self.alpha = None
        self.blocks = []
        self.blocks.append(ProgGANDiscBlock(None, initialiser, weight_const))

        for i in range(4):
            new_block = ProgGANDiscBlock(self.blocks[i], initialiser, weight_const)
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self test on start up
        for i in range(4):
            test = tf.zeros((2, 4 * (2 ** i), 4 * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, alpha=None).shape == (2, 1, 1, 1), self.blocks[i](test).shape
        
        for i in range(4):
            test = tf.zeros((2, 4 * (2 ** i), 4 * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, alpha=0.5).shape == (2, 1, 1, 1), self.blocks[i](test, alpha=0.5).shape

    def call(self, x, scale, training=True):
        self.blocks[scale].trainable = True
        x = self.blocks[scale](x, self.alpha)

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

        self.alpha = None
        self.blocks = []
        self.blocks.append(ProgGANGenBlock(latent_dims, None, initialiser, weight_const))

        for i in range(4):
            new_block = ProgGANGenBlock(latent_dims, self.blocks[i], initialiser, weight_const)
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self test on start up
        for i in range(4):
            test = tf.zeros((2, 128), dtype=tf.float32)
            assert self.blocks[i](test, alpha=None).shape == (2, 4 * (2 ** i), 4 * (2 ** i), 3), self.blocks[i](test, alpha=None).shape

        for i in range(4):
            test = tf.zeros((2, 128), dtype=tf.float32)
            assert self.blocks[i](test, alpha=0.5).shape == (2, 4 * (2 ** i), 4 * (2 ** i), 3), self.blocks[i](test, alpha=0.5).shape

    def call(self, x, scale, training=True):
        self.blocks[scale].trainable = True
        x = self.blocks[scale](x, self.alpha)

        return tf.nn.tanh(x)

