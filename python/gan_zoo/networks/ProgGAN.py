import numpy as np
import tensorflow as tf

from .blocks.progganblocks import ProgGANGeneratorFirstBlock, ProGANGeneratorLaterBlock
from .sharedarchitecture.progressiveblocks import ProgressiveDiscriminatorBlock
from .sharedarchitecture.progressivemodel import ProgressiveBase, ProgressiveGeneratorBase, ProgressiveDiscriminatorBase


class ProgressiveGAN(ProgressiveBase):

    def __init__(self, config):
        super().__init__(config)

        self.Generator = ProgGANGenerator(config=config)
        self.Discriminator = ProgGANDiscriminator(config=config)

        # Exponential moving average of generator weights for images
        self.EMAGenerator = ProgGANGenerator(config=config, name="EMAGenerator")
        self.update_mvag_generator(initial=True)

    # @tf.function
    def train_step(self, real_images):

        if self.fade_iter:
            self.Discriminator.alpha = self.fade_count / self.fade_iter
            self.Generator.alpha = self.fade_count / self.fade_iter

        else:
            self.Discriminator.alpha = None
            self.Generator.alpha = None

        #TODO set scales
        self.discriminator_step(real_images)
        self.generator_step()

        # Update MVAG and fade count
        self.update_mvag_generator()
        self.fade_count += 1

        return {"d_loss": self.d_metric.result(), "g_loss": self.g_metric.result()}
    
    def call(self, num_examples: int = 0):
        if num_examples == 0:
            imgs = self.Generator(self.fixed_noise, training=False)
        
        else:
            latent_noise = tf.random.normal((num_examples, self.latent_dims), dtype=tf.float32)
            imgs = self.Generator(latent_noise, training=False)

        return imgs


#-------------------------------------------------------------------------
""" Generator class for ProgressiveGAN """

class ProgGANGenerator(ProgressiveGeneratorBase):

    def __init__(self, config, name="Generator"):
        super().__init__(config, name)
        self.build_network(ProgGANGeneratorFirstBlock, ProGANGeneratorLaterBlock, config)

    def call(self, z, training=True):
        _, rgb = self.blocks[self.scale](z, fade_alpha=self.alpha)

        if self.output_activation == "tanh":
            return tf.nn.tanh(rgb)
        else:
            return rgb


#-------------------------------------------------------------------------
""" Discriminator class for Progressive GAN """

class ProgGANDiscriminator(ProgressiveDiscriminatorBase):

    def __init__(self, config, name="Discriminator"):
        super().__init__(config, name=name)
        self.build_network(ProgressiveDiscriminatorBlock)

    def call(self, x, training=True):
        x = self.blocks[self.scale](x, self.alpha)
        
        return tf.squeeze(x)
