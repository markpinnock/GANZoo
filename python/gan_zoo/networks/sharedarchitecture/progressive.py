import abc
import numpy as np
import tensorflow as tf

from ..model import BaseGAN


class ProgressiveBase(BaseGAN, abc.ABC):

    @abc.abstractmethod
    def __init__(self, config):
        super().__init__(config)

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
        self.Generator.mb_size = mb_size
        self.EMAGenerator.mb_size = mb_size
        self.Discriminator.mb_size = mb_size
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
                new_weights = self.EMA_beta * self.EMAGenerator.trainable_weights[idx] + (1 - self.EMA_beta) * self.Generator.trainable_weights[idx]
                self.EMAGenerator.trainable_weights[idx].assign(new_weights)
    
    @abc.abstractmethod
    def train_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def call(self):
        raise NotImplementedError


#-------------------------------------------------------------------------

class ProgressiveGeneratorBase(tf.keras.layers.Layer, abc.ABC):

    def __init__(self, config, name):
        super().__init__(name)
        self.config = config
        self.alpha = None # TODO: Check init in progressive.py 
        self.scale = None

        self.num_layers = int(np.log2(config["MAX_RES"])) - int(np.log2(config["START_RES"])) + 1
        self.resolutions = [config["START_RES"] * 2 ** idx for idx in range(self.num_layers)]
        self.latent_dims = config["LATENT_DIM"]
        self.output_activation = config["G_OUT"]
        assert self.output_activation in ["tanh", "linear"], "Choose tanh or linear output"
    
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

    @abc.abstractmethod
    def call(self):
        raise NotImplementedError


#-------------------------------------------------------------------------

class ProgressiveDiscriminatorBase(tf.keras.layers.Layer):

    def __init__(self, config, name="Discriminator"):
        super().__init__(config, name=name)
        self.config = config
        self.alpha = None
        self.scale = None
        self.num_layers = int(np.log2(config["MAX_RES"])) - int(np.log2(config["START_RES"])) + 1
        self.resolutions = [self.config["START_RES"] * 2 ** idx for idx in range(self.num_layers)]

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

    @abc.abstractmethod
    def call(self):
        raise NotImplementedError