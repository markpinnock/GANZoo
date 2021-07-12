import numpy as np
import tensorflow as tf


from ..sharedarchitecture.progressiveblocks import EqLrDense, EqLrConv2D, fade_in, pixel_norm

#-------------------------------------------------------------------------
""" First ProGAN generator block used for lowest res """

class ProgGANGeneratorFirstBlock(tf.keras.layers.Layer):
    def __init__(self, ch, res, config, name=None):
        super().__init__(name=name)
        initialiser = tf.keras.initializers.RandomNormal(0, 1)

        # Dense latent noise mapping and initial convolutional layers
        self.dense = EqLrDense(units=res * res * config["LATENT_DIM"], kernel_initializer=initialiser, name="dense")
        self.reshape = tf.keras.layers.Reshape((res, res, config["LATENT_DIM"]))
        self.conv = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv")
        self.to_rgb = EqLrConv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

    def call(self, z, fade_alpha=None):
        x = self.dense(z, noise=None, gain=tf.sqrt(2.0) / 4) # As in original implementation
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = pixel_norm(x)
        x = self.reshape(x)

        x = self.conv(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = pixel_norm(x)
        rgb = self.to_rgb(x)
        
        return x, rgb


#-------------------------------------------------------------------------
""" Basic ProGAN Generator block used after ProGenFirstBlock"""

class ProGANGeneratorLaterBlock(tf.keras.layers.Layer):
    def __init__(self, ch, res, prev_block, config, name=None):
        super().__init__(name=name)
        initialiser = tf.keras.initializers.RandomNormal(0, 1)
        
        # Previous (lower resolution) block
        self.prev_block = prev_block
        
        # Up-sampling and convolutional layers
        self.upsample = tf.keras.layers.UpSampling2D(interpolation="bilinear", name="up2D")
        self.conv1 = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv1")
        self.conv2 = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv2")
        self.to_rgb = EqLrConv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

    def call(self, z, fade_alpha=None):
        
        # Get previous non-RGB and RGB outputs
        prev_x, prev_rgb = self.prev_block(z, fade_alpha=None)
        
        # Up-sample and perform convolution on non-RGB
        prev_x = self.upsample(prev_x)
        x = pixel_norm(tf.nn.leaky_relu(self.conv1(prev_x), alpha=0.2)) # TODO noise = None
        x = pixel_norm(tf.nn.leaky_relu(self.conv2(x), alpha=0.2)) # TODO noise = None
        rgb = self.to_rgb(x) # TODO noise = None

        # If fade in, merge previous block's RGB and this block's RGB
        if fade_alpha != None:
            prev_rgb = self.upsample(prev_rgb)
            rgb = fade_in(fade_alpha, prev_rgb, rgb)
        
        return x, rgb
