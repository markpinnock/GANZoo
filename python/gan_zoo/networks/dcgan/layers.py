import tensorflow as tf


"""
Radford et al. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016.
https://arxiv.org/abs/1511.06434
"""


#-------------------------------------------------------------------------
""" DCGAN Dense block (wrapper to allow leaky relu) """
class DCDense(tf.keras.layers.Dense):

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

class DownBlock(tf.keras.layers.Layer):

    def __init__(self, config, channels, init, batchnorm=True, final=False, name=None):
        super().__init__(name=name)
        assert config["D_ACT"] in ["relu", "lrelu"]
        self.activation = config["D_ACT"]

        if final:
            self.conv = tf.keras.layers.Conv2D(filters=channels, kernel_size=(4, 4), strides=(1, 1), padding="VALID", kernel_initializer=init, name="conv")
        else:
            self.conv = tf.keras.layers.Conv2D(filters=channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME", kernel_initializer=init, name="conv")
        
        if batchnorm:
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")
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

class UpBlock(tf.keras.layers.Layer):

    def __init__(self, config, channels, init, batchnorm=True, first=False, name=None):
        super().__init__(name=name)
        assert config["G_ACT"] in ["relu", "lrelu"]
        self.activation = config["G_ACT"]

        if first:
            self.conv = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=(4, 4), strides=(1, 1), padding="VALID", kernel_initializer=init, name="conv")
        else:
            self.conv = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME", kernel_initializer=init, name="conv")
        
        if batchnorm:
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")
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
