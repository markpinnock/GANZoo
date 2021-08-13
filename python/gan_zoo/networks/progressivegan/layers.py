import numpy as np
import tensorflow as tf


#-------------------------------------------------------------------------
""" Overloaded implementation of Dense layer for equalised learning rate,
    taken from https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/core.py """

class EqLrDense(tf.keras.layers.Dense):

    def __init__(self, gain=tf.sqrt(2.0), **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
    
    def build(self, input_shape):
        super().build(input_shape)
        fan_in = tf.shape(self.kernel)[0]
        self.wscale = self.gain / tf.sqrt(tf.cast(fan_in, tf.float32))
    
    def call(self, x, lr_mul=1.0):

        """ Overloaded call to apply weight scale at runtime """

        x = tf.matmul(x, self.kernel * self.wscale * lr_mul)
        x = tf.add(x, self.bias)

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
        y = tf.tile(y, [group_size, dims[1], dims[2], 1])
    
        return tf.concat([x, y], axis=-1, name=scope)


#-------------------------------------------------------------------------
""" Pixel normalisation from ProgGAN and StyleGAN mapping network """

def pixel_norm(x):
    with tf.name_scope("pixel_norm") as scope:
        x_sq = tf.reduce_mean(tf.square(x, name=f"{scope}_square"), axis=-1, keepdims=True, name=f"{scope}_mean")
        x_norm = tf.sqrt(x_sq + 1e-8, name=f"{scope}_sqrt")
    
        return x / x_norm
