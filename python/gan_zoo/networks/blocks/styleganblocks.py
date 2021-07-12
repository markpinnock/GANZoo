import numpy as np
import tensorflow as tf

from ..sharedarchitecture.progressiveblocks import EqLrDense, EqLrConv2D, fade_in, pixel_norm


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
""" Basic StyleGAN Generator block used after StyleGenFirstBlock"""

class StyleGANGeneratorLaterBlock(tf.keras.layers.Layer):
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

class StyleGANGeneratorFirstBlock(tf.keras.layers.Layer):
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


#-------------------------------------------------------------------------
""" Instance normalisation from StyleGAN """

def instance_norm(x):
    with tf.name_scope("instance_norm") as scope:
        x_mu = tf.reduce_mean(x, axis=[1, 2], keepdims=True, name="mean")
        x_sig = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
        x = (x - x_mu) / (x_sig + 1e-8)

        return x

#-------------------------------------------------------------------------
""" Style modulation layer from StyleGAN - maps latent W to
    affine transforms for each generator block """

class StyleModulation(tf.keras.layers.Layer):
    def __init__(self, nf, name=None):
        super().__init__(name=name)
        self.nf = nf
        self.dense = EqLrDense(units=nf * 2, kernel_initializer=keras.initializers.RandomNormal(0, 1), name="dense")
    
    def call(self, x, w):
        """ x: feature maps from conv stack, w: latent vector """

        w = self.dense(w, gain=1.0)
        w = tf.reshape(w, [-1, 2, 1, 1, self.nf])

        # Style components
        ys = w[:, 0, :, :, :] + 1 # I.e. initialise bias to 1
        yb = w[:, 1, :, :, :]

        return x * ys + yb

#-------------------------------------------------------------------------
""" Additive noise layer from StyleGAN """

class AdditiveNoise(tf.keras.layers.Layer):

    """ nf: number of feature maps in corresponding generator block """

    def __init__(self, nf, name=None):
        super().__init__(name=name)
        self.nf = nf
        self.noise_weight = self.add_weight(name=f"{self.name}/noise_weight", shape=[1, 1, 1, nf], initializer="zeros", trainable=True)
    
    def call(self, x):
        NHW = x.shape[0:3]
        noise = tf.random.normal(NHW + [1])

        return x + self.noise_weight * noise
