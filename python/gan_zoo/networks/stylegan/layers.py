import numpy as np
import tensorflow as tf


#-------------------------------------------------------------------------
""" Overloaded implementation of Dense layer for equalised learning rate,
    taken from https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/core.py """

class EqLrDense(tf.keras.layers.Dense):

    def __init__(self, gain=tf.sqrt(2.0), lr_mul=1.0, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.lr_mul = lr_mul
    
    def build(self, input_shape):
        super().build(input_shape)
        fan_in = tf.shape(self.kernel)[0]
        self.wscale = self.gain / tf.sqrt(tf.cast(fan_in, tf.float32))
        """" CHECK INIT """
    def call(self, x):

        """ Overloaded call to apply weight scale at runtime """

        # Perform dense layer matmul and add noise
        x = tf.matmul(x, self.kernel * self.wscale * self.lr_mul)
        x = tf.add(x, self.bias * self.lr_mul)

        return x


#-------------------------------------------------------------------------
""" Overloaded implementation of Conv2D layer for equalised learning rate,
    taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py """

class EqLrConv2D(tf.keras.layers.Conv2D):

    def __init__(self, gain=tf.sqrt(2.0), **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
    
    def build(self, input_shape):
        super().build(input_shape)
        fan_in = tf.reduce_prod(tf.shape(self.kernel)[:-1])
        self.wscale = self.gain / tf.sqrt(tf.cast(fan_in, tf.float32))
        
    def call(self, x):

        """ Overloaded call method applies weight scale at runtime """

        # Perform convolution and add bias weights (optional noise step for StyleGAN)
        x = self._convolution_op(x, self.kernel * self.wscale)
        x = tf.add(x, self.bias)

        return x


#-------------------------------------------------------------------------
""" Fade in from ProgGAN and StyleGAN
    - interpolates between two layers by factor alpha """

def fade_in(alpha, old, new):
    with tf.name_scope("fade_in") as scope:
        return (1.0 - alpha) * old + alpha * new


#-------------------------------------------------------------------------
""" Minibatch standard deviation from ProgGAN and StyleGAN """

def mb_stddev(x, group_size=4):
    with tf.name_scope("mb_stddev") as scope:
        dims = tf.shape(x)
        group_size = tf.reduce_min([group_size, dims[0]])
        y = tf.reshape(x, [group_size, -1, dims[1], dims[2], dims[3]])
        y = tf.reduce_mean(tf.math.reduce_std(y, axis=0), axis=[1, 2, 3], keepdims=True)
        """ Check average over channels """
        y = tf.tile(y, [group_size, dims[1], dims[2], 1])
    
        return tf.concat([x, y], axis=-1, name=scope)


#-------------------------------------------------------------------------
""" Pixel normalisation from ProgGAN and StyleGAN mapping network """

def pixel_norm(x):
    with tf.name_scope("pixel_norm") as scope:
        x_sq = tf.reduce_mean(tf.square(x, name=f"{scope}_square"), axis=-1, keepdims=True, name=f"{scope}_mean")
        x_norm = tf.sqrt(x_sq + 1e-8, name=f"{scope}_sqrt")
    
        return x / x_norm


#-------------------------------------------------------------------------
""" Instance normalisation from StyleGAN """

def instance_norm(x):
    with tf.name_scope("instance_norm") as scope:
        x_mu = tf.reduce_mean(x, axis=[1, 2], keepdims=True, name="mean")
        x_sig = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
        x = (x - x_mu) / (x_sig + 1e-8)

        return x

#-------------------------------------------------------------------------
""" Style modulation layer from StyleGAN - maps latent W to
    affine transforms for each generator block """

class StyleModulation(tf.keras.layers.Layer):
    def __init__(self, nf, name=None):
        super().__init__(name=name)
        self.nf = nf
        self.dense = EqLrDense(gain=1.0, units=nf * 2, kernel_initializer=tf.keras.initializers.RandomNormal(0, 1), name="dense")

    def call(self, x, w):
        """ x: feature maps from conv stack, w: latent vector """

        w = self.dense(w)
        w = tf.reshape(w, [-1, 2, 1, 1, self.nf])

        # Style components
        ys = w[:, 0, :, :, :] + 1 # I.e. initialise bias to 1
        yb = w[:, 1, :, :, :]

        return x * ys + yb

#-------------------------------------------------------------------------
""" Additive noise layer from StyleGAN """

class AdditiveNoise(tf.keras.layers.Layer):

    """ nf: number of feature maps in corresponding generator block """

    def __init__(self, nf, name=None):
        super().__init__(name=name)
        self.nf = nf
        self.noise_weight = self.add_weight(name=f"{self.name}/noise_weight", shape=[1, 1, 1, nf], initializer="zeros", trainable=True)
    
    def call(self, x):
        NHW = x.shape[0:3]
        noise = tf.random.normal(NHW + [1])

        return x + self.noise_weight * noise
