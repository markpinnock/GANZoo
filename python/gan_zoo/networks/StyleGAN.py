from gan_zoo.networks.blocks import GANDiscBlock
import numpy as np
import tensorflow as tf

from .blocks import StyleGenFirstBlock, StyleGenLaterBlock, ProStyleGANDiscriminator
from sharedarchitecture.progressive import ProgressiveBase, ProgressiveGeneratorBase, ProgressiveDiscriminatorBase


#-------------------------------------------------------------------------

class StyleGAN(ProgressiveBase):

    def __init__(self, config):
        super().__init__(config)

        self.Generator = StyleGANGenerator(config=config)
        self.Discriminator = StyleGANDiscriminator(config=config)

        # Exponential moving average of generator weights for images
        self.EMAGenerator = StyleGANGenerator(config=config, name="EMAGenerator")
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


#-------------------------------------------------------------------------
""" Generator class for StyleGAN """

class StyleGANGenerator(ProgressiveGeneratorBase):

    def __init__(self, config, name="Generator"):
        super().__init__(config, name)
        self.StyleMap = None
        self.build_network(StyleGenFirstBlock, StyleGenLaterBlock, config)

    def build_network(self, FirstBlock, LaterBlock, config):
        super().build_network(FirstBlock, LaterBlock, config)

        self.StyleMap = MappingNet(config["STYLE_MAP_UNITS"], config["LATENT_DIM"], config["STYLE_MAP_LAYERS"], name="StyleMapping")
        _ = self.StyleMap(tf.zeros((2, self.latent_dims))) # Build implicitly until build method defined

    def call(self, z, scale, training=True):
        w = self.StyleMap(z)
        _, rgb = self.blocks[scale](w, fade_alpha=self.alpha)

        if self.output_activation == "tanh":
            return tf.nn.tanh(rgb)
        else:
            return rgb


#-------------------------------------------------------------------------
""" Discriminator class for StyleGAN """

class StyleGANDiscriminator(ProgressiveDiscriminatorBase):

    def __init__(self, config, name=None):
        super().__init__(config, name="Discriminator")
        self.build_network(GANDiscBlock)

    def call(self, x, training=True):
        x = self.blocks[self.scale](x, self.alpha)
        
        return tf.squeeze(x)
