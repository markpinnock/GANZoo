import tensorflow as tf

from .model import BaseGAN


class DCGAN(BaseGAN):

    def __init__(self, config):
        super().__init__(config)
        self.Generator = DCGANGenerator(config)
        self.Discriminator = DCGANDiscriminator(config)
    
    @tf.function
    def train_step(self, data):
        self.discriminator_step(real_images=data)
        self.generator_step()


class DCGANGenerator(tf.keras.layers.Layer):

    def __init__(self, config, name="Generator"):
        super().__init__(name=name)


class DCGANDiscriminator(tf.keras.layers.Layer):

    def __init__(self, config, name="Discriminator"):
        super().__init__(name=name)