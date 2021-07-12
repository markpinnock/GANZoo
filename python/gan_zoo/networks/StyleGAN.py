import numpy as np
import tensorflow as tf

from .blocks.styleganblocks import StyleGANGeneratorFirstBlock, StyleGANGeneratorLaterBlock, MappingNet
from .sharedarchitecture.progressiveblocks import ProgressiveDiscriminatorBlock
from .sharedarchitecture.progressivemodel import ProgressiveBase, ProgressiveGeneratorBase, ProgressiveDiscriminatorBase

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

        return {"d_loss": self.d_metric.result(), "g_loss": self.g_metric.result()}
    
    def call(self, num_examples: int = None, training: bool = False):
        if num_examples == 0:
            imgs = self.Generator(self.fixed_noise, training=training)
        
        else:
            latent_noise = tf.random.normal((num_examples, self.latent_dims), dtype=tf.float32)
            imgs = self.Generator(latent_noise, training=training)

        return imgs


#-------------------------------------------------------------------------
""" Generator class for StyleGAN """

class StyleGANGenerator(ProgressiveGeneratorBase):

    def __init__(self, config, name="Generator"):
        super().__init__(config, name)
        self.StyleMap = None
        self.build_network(StyleGANGeneratorFirstBlock, StyleGANGeneratorLaterBlock, config)

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
        self.build_network(ProgressiveDiscriminatorBlock)

    def call(self, x, training=True):
        x = self.blocks[self.scale](x, self.alpha)
        
        return tf.squeeze(x)
