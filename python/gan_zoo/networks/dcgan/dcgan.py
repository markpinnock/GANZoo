import numpy as np
import tensorflow as tf
from abc import abstractmethod

from .layers import DCDense, DownBlock, UpBlock
from networks.model import BaseGAN
from utils.losses import gradient_penalty


"""
Radford et al. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016.
https://arxiv.org/abs/1511.06434
"""


#-------------------------------------------------------------------------

class DCGAN(BaseGAN):

    def __init__(self, config):
        super().__init__(config)
        self.mb_size = config["MB_SIZE"]
        self.Generator = Generator(config=config)
        self.Discriminator = Discriminator(config=config)

    @tf.function
    def train_step(self, real_images):
        self.discriminator_step(real_images)
        self.generator_step()

        return {"d_loss": self.d_metric.result(), "g_loss": self.g_metric.result()}
    
    def call(self, num_examples: int = 0):
        if num_examples == 0:
            imgs = self.Generator(self.fixed_noise)
        
        else:
            latent_noise = tf.random.normal((num_examples, self.latent_dims), dtype=tf.float32)
            imgs = self.Generator(latent_noise)

        return imgs


#-------------------------------------------------------------------------
""" Discriminator class for DCGAN """

class Discriminator(tf.keras.Model):

    def __init__(self, config, name=None):
        super().__init__(name=name)
        init = tf.keras.initializers.RandomNormal(0, 0.02)
        self.blocks = []
        self.max_resolution = config["MAX_RES"]
        self.start_resolution = 4
        self.num_layers = int(np.log2(self.max_resolution)) - int(np.log2(self.start_resolution)) + 1
        self.channels = [np.min([(config["NDF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]

        self.blocks.append(DownBlock(config, self.channels[0], init=init, batchnorm=False, name="dn0"))

        for i in range(1, self.num_layers - 1):
            self.blocks.append(DownBlock(config, self.channels[i],  init=init, name=f"dn{i}"))

        if config["D_DENSE"]:
            self.blocks.append(tf.keras.layers.Flatten())
            self.blocks.append(DCDense(activation=config["D_ACT"], units=config["D_DENSE_UNITS"]), kernel_initializer=init, name="dense")

        else:
            self.blocks.append(DownBlock(config, 1, init=init, final=True, name="dn_final"))

    def apply_WGAN_GP(self, real_img, fake_img):
        epsilon = tf.random.uniform([fake_img.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * real_img + (1 - epsilon) * fake_img

        with tf.GradientTape() as tape:
            tape.watch(x_hat)
            D_hat = self(x_hat, training=True)
        
        gradients = tape.gradient(D_hat, x_hat)
        grad_penalty = gradient_penalty(gradients)

        return 10 * grad_penalty

    def call(self, x, training=True):
        for block in self.blocks:
            x = block(x)

        # x = self.out(x)

        return x
    
    def summary(self):
        x = tf.keras.layers.Input([self.max_resolution, self.max_resolution, 3])
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name="Discriminator").summary()


#-------------------------------------------------------------------------
""" Generator class for DCGAN """

class Generator(tf.keras.Model):

    def __init__(self, config, name=None):
        super().__init__(name)
        init = tf.keras.initializers.RandomNormal(0, 0.02)
        self.blocks = []
        self.latent_dims = config["LATENT_DIM"]
        self.output_activation = config["G_OUT"]
        self.max_resolution = config["MAX_RES"]
        self.start_resolution = 4
        self.num_layers = int(np.log2(self.max_resolution)) - int(np.log2(self.start_resolution)) + 1
        assert self.output_activation in ["tanh", "linear"], "Choose tanh or linear output"
        self.channels = [np.min([(config["NGF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()

        if config["G_DENSE"]:
            dense_units = self.start_resolution * self.start_resolution * self.latent_dims
            self.blocks.append(DCDense(activation=config["G_ACT"], units=dense_units), kernel_initializer=init, name="dense")
            self.blocks.append(tf.keras.layers.Reshape([self.start_resolution, self.start_resolution, self.latent_dims]))

        else:
            self.blocks.append(tf.keras.layers.Reshape([1, 1, self.latent_dims]))
            self.blocks.append(UpBlock(config, self.channels[1], init=init, first=True, name="up0"))

        for i in range(1, self.num_layers - 1):
            self.blocks.append(UpBlock(config, self.channels[i + 1],  init=init, name=f"up{i}"))
        
        self.blocks.append(UpBlock(config, 3, init=init, batchnorm=False, name="up_final"))

    def call(self, x, training=True):
        for block in self.blocks:
            x = block(x)

        if self.output_activation == "tanh":
            return tf.nn.tanh(x)
        else:
            return x
    
    def summary(self):
        x = tf.keras.layers.Input([self.latent_dims])
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name="Generator").summary()
