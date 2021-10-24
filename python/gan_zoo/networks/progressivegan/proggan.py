import numpy as np
import tensorflow as tf

from networks.progressivegan.blocks import GeneratorFirstBlock, GeneratorLaterBlock, DiscriminatorBlock
from networks.model import BaseGAN
from utils.losses import gradient_penalty


""" Based on:
    - Karras et al. Progressive Growing of GANs for Improved Quality, Stability, and Variation
    - https://arxiv.org/abs/1710.10196
    - https://github.com/tkarras/progressive_growing_of_gans """


#-------------------------------------------------------------------------

class ProgressiveGAN(BaseGAN):

    def __init__(self, config):
        super().__init__(config)

        self.Generator = Generator(config=config)
        self.Discriminator = Discriminator(config=config)

        # Exponential moving average of generator weights for images
        self.EMAGenerator = Generator(config=config, name="EMAGenerator")
        self.update_mvag_generator(initial=True)

        self.EMA_beta = config["EMA_BETA"]
        self.fade_iter = 0
        self.fade_count = 0
        self.mb_size = None
        self.alpha = None

    def fade_set(self, num_iter):
        """ Activates or deactivates fade in """

        self.fade_iter = num_iter
        self.fade_count = 0

    def set_scale(self, scale, mb_size):
        """ Sets new block to trainable and sets to_rgb/from_rgb
            conv layers in old blocks to untrainable
            to avoid missing gradients warning """

        self.Generator.scale = scale
        self.EMAGenerator.scale = scale
        self.Discriminator.scale = scale
        self.mb_size = mb_size

        self.Discriminator.blocks[scale].trainable = True
       
        for i in range(0, scale):
            self.Discriminator.blocks[i].from_rgb.trainable = False
        
        self.Generator.blocks[scale].trainable = True
        
        for i in range(0, scale):
            self.Generator.blocks[i].to_rgb.trainable = False

        self.EMAGenerator.blocks[scale].trainable = True
        
        for i in range(0, scale):
            self.EMAGenerator.blocks[i].to_rgb.trainable = False

    def update_mvag_generator(self, initial=False):

        """ Updates EMAGenerator with Generator weights """

        # If first use, clone Generator
        if initial:
            assert len(self.Generator.weights) == len(self.EMAGenerator.weights)

            for idx in range(len(self.EMAGenerator.weights)):
                assert self.EMAGenerator.weights[idx].name == self.Generator.weights[idx].name
                self.EMAGenerator.weights[idx].assign(self.Generator.weights[idx])
            
        else:
            for idx in range(len(self.EMAGenerator.trainable_weights)):
                assert self.EMAGenerator.trainable_weights[idx].name == self.Generator.trainable_weights[idx].name
                new_weights = self.EMA_beta * self.EMAGenerator.trainable_weights[idx] + (1 - self.EMA_beta) * self.Generator.trainable_weights[idx]
                self.EMAGenerator.trainable_weights[idx].assign(new_weights)

    def train_step(self, real_images):

        if self.fade_iter:
            self.Discriminator.alpha = self.fade_count / self.fade_iter
            self.Generator.alpha = self.fade_count / self.fade_iter

        else:
            self.Discriminator.alpha = None
            self.Generator.alpha = None

        self.discriminator_step(real_images)
        self.generator_step()

        # Update MVAG and fade count
        self.update_mvag_generator()
        self.fade_count += 1

        return {"d_loss": self.d_metric.result(), "g_loss": self.g_metric.result()}
    
    def call(self, num_examples: int = 0):
        if num_examples == 0:
            imgs = self.EMAGenerator(self.fixed_noise, training=False)
        
        else:
            latent_noise = tf.random.normal((num_examples, self.latent_dims), dtype=tf.float32)
            imgs = self.EMAGenerator(latent_noise, training=False)

        return imgs


#-------------------------------------------------------------------------
""" Generator class for ProgressiveGAN """

class Generator(tf.keras.layers.Layer):

    def __init__(self, config, name="Generator"):
        super().__init__(config, name)
        self.config = config
        self.alpha = None # TODO: Check init in progressive.py 
        self.scale = None

        self.num_layers = int(np.log2(config["MAX_RES"])) - int(np.log2(config["START_RES"])) + 1
        self.resolutions = [config["START_RES"] * 2 ** idx for idx in range(self.num_layers)]
        self.latent_dims = config["LATENT_DIM"]
        self.output_activation = config["G_OUT"]
        assert self.output_activation in ["tanh", "linear"], "Choose tanh or linear output"

        self.build_network(GeneratorFirstBlock, GeneratorLaterBlock, config)

    def build_network(self, FirstBlock, LaterBlock, config):
        self.blocks = []
        self.channels = [np.min([(config["NGF"] * 2 ** i), self.config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()
        self.blocks.append(FirstBlock(self.channels[0], self.resolutions[0], config, name="block0"))

        for i in range(1, self.num_layers):
            new_block = LaterBlock(self.channels[i], self.resolutions[i], self.blocks[i - 1], config, name=f"block{i}")
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self-test
        for i in range(0, self.num_layers):
            test = tf.zeros((2, self.latent_dims), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=None)[1].shape == (2, self.config["START_RES"] * (2 ** i), self.config["START_RES"] * (2 ** i), 3), self.blocks[i](test, fade_alpha=None)[1].shape

        for i in range(0, self.num_layers):
            test = tf.zeros((2, self.latent_dims), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=0.5)[1].shape == (2, self.config["START_RES"] * (2 ** i), self.config["START_RES"] * (2 ** i), 3), self.blocks[i](test, fade_alpha=0.5)[1].shape

    def call(self, z, training=True):
        _, rgb = self.blocks[self.scale](z, fade_alpha=self.alpha)

        if self.output_activation == "tanh":
            return tf.nn.tanh(rgb)
        else:
            return rgb


#-------------------------------------------------------------------------
""" Discriminator class for Progressive GAN """

class Discriminator(tf.keras.layers.Layer):

    def __init__(self, config, name="Discriminator"):
        super().__init__(config, name=name)
        self.config = config
        self.alpha = None
        self.scale = None
        self.num_layers = int(np.log2(config["MAX_RES"])) - int(np.log2(config["START_RES"])) + 1
        self.resolutions = [self.config["START_RES"] * 2 ** idx for idx in range(self.num_layers)]

        self.build_network(DiscriminatorBlock)

    def build_network(self, GANDiscBlock):
        self.blocks = []
        self.channels = [np.min([(self.config["NDF"] * 2 ** i), self.config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()
        self.blocks.append(GANDiscBlock(self.channels[0], None, self.config, name="block0"))

        for i in range(1, self.num_layers):
            new_block = GANDiscBlock(self.channels[i], self.blocks[i - 1], self.config, name=f"block{i}")
            new_block.trainable = False
            self.blocks.append(new_block)

        # Recursive self test on start up
        for i in range(self.num_layers):
            test = tf.zeros((2, self.config["START_RES"] * (2 ** i), self.config["START_RES"] * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=None).shape == (2, 1), self.blocks[i](test).shape
        
        for i in range(self.num_layers):
            test = tf.zeros((2, self.config["START_RES"] * (2 ** i), self.config["START_RES"] * (2 ** i), 3), dtype=tf.float32)
            assert self.blocks[i](test, fade_alpha=0.5).shape == (2, 1), self.blocks[i](test, fade_alpha=0.5).shape
    
    def apply_WGAN_GP(self, real_img, fake_img):
        # Prevent discriminator output from drifting too far from zero
        drift_term = tf.reduce_mean(tf.square(self(real_img)))

        epsilon = tf.random.uniform([fake_img.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * real_img + (1 - epsilon) * fake_img

        with tf.GradientTape() as tape:
            tape.watch(x_hat)
            D_hat = self(x_hat, training=True)
        
        gradients = tape.gradient(D_hat, x_hat)
        grad_penalty = gradient_penalty(gradients)

        return 10 * grad_penalty + 0.001 * drift_term

    def call(self, x, training=True):
        x = self.blocks[self.scale](x, self.alpha)
        
        return tf.squeeze(x)
