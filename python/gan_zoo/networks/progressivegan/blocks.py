import numpy as np
import tensorflow as tf

from .layers import EqLrDense, EqLrConv2D, fade_in, pixel_norm, mb_stddev


#-------------------------------------------------------------------------
""" ProgressiveGAN Discriminator block """
# TODO: separate into last and prev blocks
class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, ch, next_block, config, name=None):
        super().__init__(name=name)
        double_ch = np.min([ch * 2, config["MAX_CHANNELS"]])
        initialiser = tf.keras.initializers.RandomNormal(0, 1)

        self.next_block = next_block
        self.from_rgb = EqLrConv2D(gain=tf.sqrt(2.0), filters=ch, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="from_rgb")

        # If this is last discriminator block, collapse to prediction
        if next_block == None:
            self.conv = EqLrConv2D(gain=tf.sqrt(2.0), filters=double_ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv")
            self.flat = tf.keras.layers.Flatten(name="flatten")
            self.dense = EqLrDense(gain=tf.sqrt(2.0), units=config["LATENT_DIM"], kernel_initializer=initialiser, name="dense")
            self.out = EqLrDense(gain=1.0, units=1, kernel_initializer=initialiser, name="out")
        
        # If next blocks exist, conv and downsample
        else:
            self.conv1 = EqLrConv2D(gain=tf.sqrt(2.0), filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv1")
            self.conv2 = EqLrConv2D(gain=tf.sqrt(2.0), filters=double_ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv2")
            self.downsample = tf.keras.layers.AveragePooling2D(name="down2D")

    def call(self, x, fade_alpha=None, first_block=True):

        # If fade in, pass downsampled image into next block and cache
        if first_block and fade_alpha != None and self.next_block != None:
            next_rgb = self.downsample(x)
            next_rgb = tf.nn.leaky_relu(self.next_block.from_rgb(next_rgb), alpha=0.2)

        # If the very first block, perform 1x1 conv on rgb
        if first_block:
            x = tf.nn.leaky_relu(self.from_rgb(x), alpha=0.2)

        # If this is not the last block
        if self.next_block != None:
            x = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)
            x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)
            x = self.downsample(x)

            # If fade in, merge with cached layer
            if first_block and fade_alpha != None and self.next_block != None:
                x = fade_in(fade_alpha, next_rgb, x)
            
            x = self.next_block(x, fade_alpha=None, first_block=False)
        
        # If this is the last block
        else:
            x = mb_stddev(x)
            x = tf.nn.leaky_relu(self.conv(x), alpha=0.2)
            x = self.flat(x)
            x = tf.nn.leaky_relu(self.dense(x))
            x = self.out(x)

        return x


#-------------------------------------------------------------------------
""" First ProgressiveGAN generator block used for lowest res """

class GeneratorFirstBlock(tf.keras.layers.Layer):
    def __init__(self, ch, res, config, name=None):
        super().__init__(name=name)
        initialiser = tf.keras.initializers.RandomNormal(0, 1)

        # Dense latent noise mapping and initial convolutional layers
        self.dense = EqLrDense(gain=tf.sqrt(2.0) / 4, units=res * res * config["LATENT_DIM"], kernel_initializer=initialiser, name="dense")
        self.reshape = tf.keras.layers.Reshape((res, res, config["LATENT_DIM"]))
        self.conv = EqLrConv2D(gain=tf.sqrt(2.0), filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv")
        self.to_rgb = EqLrConv2D(gain=tf.sqrt(2.0), filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

    def call(self, z, fade_alpha=None):
        x = pixel_norm(z)
        x = self.dense(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = pixel_norm(x)
        x = self.reshape(x)
        x = self.conv(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = pixel_norm(x)
        rgb = self.to_rgb(x)
        
        return x, rgb


#-------------------------------------------------------------------------
""" Basic ProgressiveGAN Generator block used after ProGenFirstBlock"""

class GeneratorLaterBlock(tf.keras.layers.Layer):
    def __init__(self, ch, res, prev_block, config, name=None):
        super().__init__(name=name)
        initialiser = tf.keras.initializers.RandomNormal(0, 1)
        
        # Previous (lower resolution) block
        self.prev_block = prev_block
        
        # Up-sampling and convolutional layers
        self.upsample = tf.keras.layers.UpSampling2D(interpolation="bilinear", name="up2D")
        self.conv1 = EqLrConv2D(gain=tf.sqrt(2.0), filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv1")
        self.conv2 = EqLrConv2D(gain=tf.sqrt(2.0), filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv2")
        self.to_rgb = EqLrConv2D(gain=tf.sqrt(2.0), filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

    def call(self, z, fade_alpha=None):
        
        # Get previous non-RGB and RGB outputs
        prev_x, prev_rgb = self.prev_block(z, fade_alpha=None)
        
        # Up-sample and perform convolution on non-RGB
        prev_x = self.upsample(prev_x)
        x = pixel_norm(tf.nn.leaky_relu(self.conv1(prev_x), alpha=0.2))
        x = pixel_norm(tf.nn.leaky_relu(self.conv2(x), alpha=0.2))
        rgb = self.to_rgb(x)

        # If fade in, merge previous block's RGB and this block's RGB
        if fade_alpha is not None:
            prev_rgb = self.upsample(prev_rgb)
            rgb = fade_in(fade_alpha, prev_rgb, rgb)
        
        return x, rgb
