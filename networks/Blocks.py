import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from .Layers import (
    fade_in,
    mb_stddev,
    pixel_norm,
    instance_norm,
    AdditiveNoise,
    StyleModulation,
    EqLrDense,
    EqLrConv2D
    )


#-------------------------------------------------------------------------
""" Style mapping network """

class MappingNet(keras.layers.Layer):
    
    def __init__(self, num_units, latent_dim, num_layers, name=None):
        super().__init__(name=name)

        self.lr_mul = 0.01
        std_init = 1 / self.lr_mul
        self.dense = [EqLrDense(units=num_units, kernel_initializer=keras.initializers.RandomNormal(0, std_init), name=f"dense_{i}") for i in range(num_layers - 1)]
        self.dense.append(EqLrDense(units=latent_dim, kernel_initializer=keras.initializers.RandomNormal(0, std_init), name=f"dense_{num_layers - 1}"))

    def call(self, z):
        w = pixel_norm(z)

        for layer in self.dense:
            w = tf.nn.leaky_relu(layer(w, noise=None, gain=tf.sqrt(2.0), lr_mul=self.lr_mul))

        return w

#-------------------------------------------------------------------------
""" Prog- and StyleGAN Discriminator block """
# TODO: separate into last and prev blocks
class GANDiscBlock(keras.layers.Layer):
    def __init__(self, ch, next_block, config, name=None):
        super().__init__(name=name)

        double_ch = np.min([ch * 2, config["MAX_CHANNELS"]])
        initialiser = keras.initializers.RandomNormal(0, 1)

        self.next_block = next_block
        self.from_rgb = EqLrConv2D(filters=ch, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="from_rgb")

        # If this is last discriminator block, collapse to prediction
        if next_block == None:
            self.conv = EqLrConv2D(filters=double_ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv")
            self.flat = keras.layers.Flatten(name="flatten")
            self.dense = EqLrDense(units=config["LATENT_DIM"], kernel_initializer=initialiser, name="dense")
            self.out = EqLrDense(units=1, kernel_initializer=initialiser, name="out")
        
        # If next blocks exist, conv and downsample
        else:
            self.conv1 = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv1")
            self.conv2 = EqLrConv2D(filters=double_ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv2")
            self.downsample = keras.layers.AveragePooling2D(name="down2D")

    def call(self, x, fade_alpha=None, first_block=True):

        # If fade in, pass downsampled image into next block and cache
        if first_block and fade_alpha != None and self.next_block != None:
            next_rgb = self.downsample(x)
            next_rgb = tf.nn.leaky_relu(self.next_block.from_rgb(next_rgb, noise=None), alpha=0.2)

        # If the very first block, perform 1x1 conv on rgb
        if first_block:
            x = tf.nn.leaky_relu(self.from_rgb(x, noise=None), alpha=0.2)

        # If this is not the last block
        if self.next_block != None:
            x = tf.nn.leaky_relu(self.conv1(x, noise=None), alpha=0.2)
            x = tf.nn.leaky_relu(self.conv2(x, noise=None), alpha=0.2)
            x = self.downsample(x)

            # If fade in, merge with cached layer
            if first_block and fade_alpha != None and self.next_block != None:
                x = fade_in(fade_alpha, next_rgb, x)
            
            x = self.next_block(x, fade_alpha=None, first_block=False)
        
        # If this is the last block
        else:
            x = mb_stddev(x)
            x = tf.nn.leaky_relu(self.conv(x, noise=None), alpha=0.2)
            x = self.flat(x)
            x = tf.nn.leaky_relu(self.dense(x, noise=None))
            x = self.out(x, noise=None, gain=1) # Gain as in original implementation

        return x

#-------------------------------------------------------------------------
""" Basic ProgGAN Generator block used after ProgGenFirstBlock"""

class ProgGenLaterBlock(keras.layers.Layer):
    def __init__(self, ch, res, prev_block, config, name=None):
        super().__init__(name=name)

        initialiser = keras.initializers.RandomNormal(0, 1)
        self.prev_block = prev_block
        
        self.upsample = keras.layers.UpSampling2D(interpolation="bilinear", name="up2D")
        self.conv1 = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv1")
        self.conv2 = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv2")
        self.to_rgb = EqLrConv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

    def call(self, z, fade_alpha=None):
        
        prev_x, prev_rgb = self.prev_block(z, fade_alpha=None)
        prev_x = self.upsample(prev_x)
        x = pixel_norm(tf.nn.leaky_relu(self.conv1(prev_x, noise=None), alpha=0.2))
        x = pixel_norm(tf.nn.leaky_relu(self.conv2(x, noise=None), alpha=0.2))
        rgb = self.to_rgb(x, noise=None)

        # If fade in, merge prev block and this block
        if fade_alpha != None:
            prev_rgb = self.upsample(prev_rgb)
            rgb = fade_in(fade_alpha, prev_rgb, rgb)
        
        return x, rgb

#-------------------------------------------------------------------------
""" Basic StyleGAN Generator block used after StyleGenFirstBlock"""

class StyleGenLaterBlock(keras.layers.Layer):
    def __init__(self, ch, res, prev_block, config, name=None):
        super().__init__(name=name)

        initialiser = keras.initializers.RandomNormal(0, 1)
        self.prev_block = prev_block
        
        self.upsample = keras.layers.UpSampling2D(interpolation="bilinear", name="up2D")
        
        self.conv1 = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv1")
        self.conv1_noise = AdditiveNoise(nf=ch, name="conv1_noise")
        self.conv1_style = StyleModulation(nf=ch, name="conv1_style")

        self.conv2 = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv2")
        self.conv2_noise = AdditiveNoise(nf=ch, name="conv2_noise")
        self.conv2_style = StyleModulation(nf=ch, name="conv2_style")

        self.to_rgb = EqLrConv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

    def call(self, w, fade_alpha=None):
        
        prev_x, prev_rgb = self.prev_block(w, fade_alpha=None)
        prev_x = self.upsample(prev_x)

        x = self.conv1(prev_x, noise=self.conv1_noise)
        x = instance_norm(tf.nn.leaky_relu(x, alpha=0.2))
        x = self.conv1_style(x, w)
    
        x = self.conv2(x, noise=self.conv2_noise)
        x = instance_norm(tf.nn.leaky_relu(x, alpha=0.2))
        x = self.conv2_style(x, w)

        rgb = self.to_rgb(x, noise=None)

        # If fade in, merge prev block and this block
        if fade_alpha != None:
            prev_rgb = self.upsample(prev_rgb)
            rgb = fade_in(fade_alpha, prev_rgb, rgb)
        
        return x, rgb

#-------------------------------------------------------------------------
""" First ProgGAN generator block used for lowest res """

class ProgGenFirstBlock(keras.layers.Layer):
    def __init__(self, ch, res, config, name=None):
        super().__init__(name=name)

        initialiser = keras.initializers.RandomNormal(0, 1)
        
        self.dense = EqLrDense(units=res * res * config["LATENT_DIM"], kernel_initializer=initialiser, name="dense")
        self.reshape = keras.layers.Reshape((res, res, config["LATENT_DIM"]))
        self.conv = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv")
        self.to_rgb = EqLrConv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

    def call(self, z, fade_alpha=None):
        x = pixel_norm(tf.nn.leaky_relu(self.dense(z, noise=None, gain=tf.sqrt(2.0) / 4), alpha=0.2)) # As in original implementation
        x = self.reshape(x)
        x = pixel_norm(tf.nn.leaky_relu(self.conv(x, noise=None), alpha=0.2))
        rgb = self.to_rgb(x, noise=None)
        
        return x, rgb

#-------------------------------------------------------------------------
""" First StyleGAN generator block used for lowest res """

class StyleGenFirstBlock(keras.layers.Layer):
    def __init__(self, ch, res, config, name=None):
        super().__init__(name=name)

        initialiser = keras.initializers.RandomNormal(0, 1)

        self.constant = self.add_weight(name=f"{name}/constant", shape=[1, res, res, config["LATENT_DIM"]], initializer="ones", trainable=True)
        self.constant_noise = AdditiveNoise(nf=config["LATENT_DIM"], name=f"{name}/const_noise")
        self.constant_style = StyleModulation(nf=config["LATENT_DIM"], name="const_style")
        
        self.conv = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv")
        self.conv_noise = AdditiveNoise(nf=ch, name=f"{name}/conv_noise")
        self.conv_style = StyleModulation(nf=ch, name="conv_style")
        
        self.to_rgb = EqLrConv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="to_rgb")

    def call(self, w, fade_alpha=None):
        x = self.constant_noise(self.constant)
        x = instance_norm(tf.nn.leaky_relu(x, alpha=0.2))
        x = self.constant_style(x, w)
        
        x = self.conv(x, noise=self.conv_noise)
        x = instance_norm(tf.nn.leaky_relu(x, alpha=0.2))
        x = self.conv_style(x, w)

        rgb = self.to_rgb(x, noise=None)
        
        return x, rgb
