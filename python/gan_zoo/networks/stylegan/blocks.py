import numpy as np
import tensorflow as tf

from .layers import EqLrDense, EqLrConv2D, fade_in, pixel_norm, mb_stddev, instance_norm, StyleModulation, AdditiveNoise


#-------------------------------------------------------------------------
""" Style mapping network """

class MappingNet(tf.keras.layers.Layer):
    
    def __init__(self, num_units, latent_dim, num_layers, name=None):
        super().__init__(name=name)
        self.lr_mul = 0.01
        std_init = 1 / self.lr_mul

        self.dense = [EqLrDense(units=num_units, kernel_initializer=tf.keras.initializers.RandomNormal(0, std_init), name=f"dense_{i}") for i in range(num_layers - 1)]
        self.dense.append(EqLrDense(units=latent_dim, kernel_initializer=tf.keras.initializers.RandomNormal(0, std_init), name=f"dense_{num_layers - 1}"))

    def call(self, z):
        w = pixel_norm(z)

        for layer in self.dense:
            w = tf.nn.leaky_relu(layer(w, noise=None, gain=tf.sqrt(2.0), lr_mul=self.lr_mul))

        return w


#-------------------------------------------------------------------------
""" StyleGAN Discriminator block """
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
            x = self.out(x, noise=None)

        return x


#-------------------------------------------------------------------------
""" Basic StyleGAN Generator block used after StyleGenFirstBlock"""

class GeneratorLaterBlock(tf.keras.layers.Layer):
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

        # Additive noise inputs
        if config["ADD_NOISE"]:
            self.conv1_noise = AdditiveNoise(nf=ch, name="conv1_noise")
            self.conv2_noise = AdditiveNoise(nf=ch, name="conv2_noise")
        
        else:
            self.conv1_noise = None
            self.conv2_noise = None
        
        # Style/Adaptive instance normalisation
        self.conv1_style = StyleModulation(nf=ch, name="conv1_style")
        self.conv2_style = StyleModulation(nf=ch, name="conv2_style")

    def call(self, w, fade_alpha=None):

        # Get previous non-RGB and RGB outputs
        prev_x, prev_rgb = self.prev_block(w, fade_alpha=None)
        prev_x = self.upsample(prev_x)

        # Up-sample and perform convolution on non-RGB, with AdaIN and additive noise if indicated
        x = self.conv1(prev_x)
        if self.conv1_noise: x = self.conv1_noise(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = instance_norm(x)
        x = self.conv1_style(x, w)
    
        x = self.conv2(x)
        if self.conv2_noise: x = self.conv2_noise(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = instance_norm(x)
        x = self.conv2_style(x, w)

        rgb = self.to_rgb(x)

        # If fade in, merge previous block's RGB and this block's RGB
        if fade_alpha != None:
            prev_rgb = self.upsample(prev_rgb)
            rgb = fade_in(fade_alpha, prev_rgb, rgb)
        
        return x, rgb


#-------------------------------------------------------------------------
""" First StyleGAN generator block used for lowest res """

class GeneratorFirstBlock(tf.keras.layers.Layer):
    def __init__(self, ch, res, config, name=None):
        super().__init__(name=name)
        initialiser = tf.keras.initializers.RandomNormal(0, 1)

        # Constant input and convolutional layers
        self.constant = self.add_weight(name="constant", shape=[1, res, res, config["LATENT_DIM"]], initializer="ones", trainable=True)
        self.conv = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv")
        self.to_rgb = EqLrConv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

        # Additive noise inputs
        if config["ADD_NOISE"]:
            self.constant_noise = AdditiveNoise(nf=config["LATENT_DIM"], name="const_noise")
            self.conv_noise = AdditiveNoise(nf=ch, name="conv_noise")

        else:
            self.constant_noise = None
            self.conv_noise = None

        # Style/Adaptive instance normalisation
        self.constant_style = StyleModulation(nf=config["LATENT_DIM"], name="const_style")
        self.conv_style = StyleModulation(nf=ch, name="conv_style")

    def call(self, w, fade_alpha=None):
        x = self.constant
        if self.constant_noise: x = self.constant_noise(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = instance_norm(x)
        x = self.constant_style(x, w)
        
        if self.conv_noise: x = self.conv_noise(self.conv(x))
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = instance_norm(x)
        x = self.conv_style(x, w)

        rgb = self.to_rgb(x)
        
        return x, rgb
