import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from abc import abstractmethod


#-------------------------------------------------------------------------
""" DCGAN Dense block (wrapper for dense to allow leaky relu) """
class DCDense(keras.layers.Dense):

    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        assert activation in ["relu", "lrelu"]
        self.activation == activation
    
    def call(self, x):
        x = super(x)

        if self.activation == "relu":
            x = tf.nn.relu(x)
        elif self.activation == "lrelu":
            x = tf.nn.leaky_relu(x, alpha=0.2)

#-------------------------------------------------------------------------
""" DCGAN Discriminator block """

class DownBlock(keras.layers.Layer):

    def __init__(self, config, channels, init, batchnorm=True, final=False, name=None):
        super().__init__(name=name)
        assert config["D_ACT"] in ["relu", "lrelu"]
        self.activation = config["D_ACT"]

        if final:
            self.conv = keras.layers.Conv2D(filters=channels, kernel_size=(4, 4), strides=(1, 1), padding="VALID", kernel_initializer=init, name="conv")
        else:
            self.conv = keras.layers.Conv2D(filters=channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME", kernel_initializer=init, name="conv")
        
        if batchnorm:
            self.bn = keras.layers.BatchNormalization(name="batchnorm")
        else:
            self.bn = None
    
    def call(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)

        if self.activation == "relu":
            x = tf.nn.relu(x)
        elif self.activation == "lrelu":
            x = tf.nn.leaky_relu(x, alpha=0.2)
        
        return x

#-------------------------------------------------------------------------
""" DCGAN Generator block """

class UpBlock(keras.layers.Layer):

    def __init__(self, config, channels, init, batchnorm=True, first=False, name=None):
        super().__init__(name=name)
        assert config["G_ACT"] in ["relu", "lrelu"]
        self.activation = config["G_ACT"]

        if first:
            self.conv = keras.layers.Conv2DTranspose(filters=channels, kernel_size=(4, 4), strides=(1, 1), padding="VALID", kernel_initializer=init, name="conv")
        else:
            self.conv = keras.layers.Conv2DTranspose(filters=channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME", kernel_initializer=init, name="conv")
        
        if batchnorm:
            self.bn = keras.layers.BatchNormalization(name="batchnorm")
        else:
            self.bn = None
    
    def call(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)

        if self.activation == "relu":
            x = tf.nn.relu(x)
        elif self.activation == "lrelu":
            x = tf.nn.leaky_relu(x, alpha=0.2)
        
        return x

#-------------------------------------------------------------------------
""" Base class for both generator and discriminator """

class BaseNetwork(keras.Model):

    def __init__(self, config, name):
        super().__init__(name=name)
        self.blocks = []
        self.max_resolution = config["MAX_RES"]
        self.start_resolution = 4
        self.num_layers = int(np.log2(self.max_resolution)) - int(np.log2(self.start_resolution)) + 1

    @abstractmethod
    def call(self):
        raise NotImplementedError

    @abstractmethod
    def summary(self):
        raise NotImplementedError

#-------------------------------------------------------------------------
""" Discriminator class for DCGAN """

class DCDiscriminator(BaseNetwork):

    def __init__(self, config, name=None):
        super().__init__(config, name=name)
        init = keras.initializers.RandomNormal(0, 0.02)
        self.channels = [np.min([(config["NDF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.max_resolution = config["MAX_RES"]

        self.blocks.append(DownBlock(config, self.channels[0], init=init, batchnorm=False, name="dn0"))

        for i in range(1, self.num_layers - 1):
            self.blocks.append(DownBlock(config, self.channels[i],  init=init, name=f"dn{i}"))

        if config["D_DENSE"]:
            self.blocks.append(keras.layers.Flatten())
            self.blocks.append(DCDense(activation=config["D_ACT"], units=config["D_DENSE_UNITS"]), kernel_initializer=init, name="dense")

        else:
            self.blocks.append(DownBlock(config, self.channels[-1], init=init, final=True, name="dn_final"))
        
        self.out = keras.layers.Dense(units=1, kernel_initializer=init, name="out")

    def call(self, x, training=True):
        for block in self.blocks:
            x = block(x)
        
        return tf.squeeze(x)
    
    def summary(self):
        x = keras.layers.Input([self.max_resolution, self.max_resolution, 3])
        return keras.Model(inputs=[x], outputs=self.call(x), name="Discriminator").summary()

#-------------------------------------------------------------------------
""" Generator class for DCGAN """

class DCGenerator(BaseNetwork):

    def __init__(self, config, name=None):
        super().__init__(config, name)
        init = keras.initializers.RandomNormal(0, 0.02)
        self.latent_dims = config["LATENT_DIM"]
        self.output_activation = config["G_OUT"]
        assert self.output_activation in ["tanh", "linear"], "Choose tanh or linear output"
        self.channels = [np.min([(config["NGF"] * 2 ** i), config["MAX_CHANNELS"]]) for i in range(self.num_layers)]
        self.channels.reverse()

        if config["G_DENSE"]:
            dense_units = self.start_resolution * self.start_resolution * self.latent_dims
            self.blocks.append(DCDense(activation=config["G_ACT"], units=dense_units), kernel_initializer=init, name="dense")
            self.blocks.append(keras.layers.Reshape([self.start_resolution, self.start_resolution, self.latent_dims]))

        else:
            self.blocks.append(keras.layers.Reshape([1, 1, self.latent_dims]))
            self.blocks.append(UpBlock(config, self.channels[1], init=init, first=True, name="up0"))

        for i in range(1, self.num_layers - 1):
            self.blocks.append(UpBlock(config, self.channels[i + 1],  init=init, name=f"up{i}"))
        
        self.blocks.append(UpBlock(config, 3, init=init, batchnorm=False, name="upf"))

    def call(self, x, training=True):
        for block in self.blocks:
            x = block(x)

        if self.output_activation == "tanh":
            return tf.nn.tanh(x)
        else:
            return x
    
    def summary(self):
        x = keras.layers.Input([self.latent_dims])
        return keras.Model(inputs=[x], outputs=self.call(x), name="Generator").summary()
