import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_math_ops, nn_ops


#-------------------------------------------------------------------------
""" Overloaded implementation of Dense layer for equalised learning rate,
    taken from https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/core.py """

class EqLrDense(keras.layers.Dense):

    def __init__(self, **kwargs):
        """ Initialise Dense with the usual arguments """

        super().__init__(**kwargs)

        self.weight_scale = None
    
    def call(self, inputs, noise=None, gain=tf.sqrt(2.0), lr_mul=1.0):
        """ Overloaded call to apply weight scale at runtime """

        if self.weight_scale is None:
            fan_in = tf.reduce_prod(self.kernel.shape[:-1])
            self.weight_scale = gain / tf.sqrt(tf.cast(fan_in, tf.float32))

        # Perform dense layer matmul (optional noise step for StyleGAN)
        outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel * self.weight_scale * lr_mul)
        if noise: outputs = noise(outputs)
        outputs = nn_ops.bias_add(outputs, self.bias * self.weight_scale * lr_mul)
        
        # Activation not needed
        return outputs

#-------------------------------------------------------------------------
""" Overloaded implementation of Conv2D layer for equalised learning rate,
    taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py """

class EqLrConv2D(keras.layers.Conv2D):

    def __init__(self, **kwargs):
        """ Initialise with the usual arguments """

        super().__init__(**kwargs)

        self.weight_scale = None
        
    def call(self, inputs, noise=None, gain=tf.sqrt(2.0)):
        """ Overloaded call method applies weight scale at runtime """

        if self.weight_scale is None: # TODO: implement in .build()
            fan_in = tf.reduce_prod(self.kernel.shape[:-1])
            self.weight_scale = gain / tf.sqrt(tf.cast(fan_in, tf.float32))

        # Perform convolution and add bias weights (optional noise step for StyleGAN)
        outputs = self._convolution_op(inputs, self.kernel * self.weight_scale)
        if noise: outputs = noise(outputs)
        outputs = tf.nn.bias_add(outputs, self.bias * self.weight_scale, data_format="NHWC")

        # Activation not needed
        return outputs

#-------------------------------------------------------------------------
""" Fade in from ProgGAN and StyleGAN
    - interpolates between two layers by factor alpha """

def fade_in(alpha, old, new):
    return (1.0 - alpha) * old + alpha * new

#-------------------------------------------------------------------------
""" Minibatch statistics from ProgGAN and StyleGAN """

def mb_stddev(x, group_size=4):
    dims = x.shape
    group_size = tf.reduce_min([group_size, dims[0]])
    y = tf.reshape(x, [group_size, -1, dims[1], dims[2], dims[3]])
    y = tf.reduce_mean(tf.math.reduce_std(y, axis=0), axis=[1, 2, 3], keepdims=True)
    y = tf.tile(y, [group_size, dims[1], dims[2], 1])
    
    return tf.concat([x, y], axis=-1)

#-------------------------------------------------------------------------
""" Pixel normalisation from ProgGAN """

def pixel_norm(x):
    x_sq = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
    x_norm = tf.sqrt(x_sq + 1e-8)
    
    return x / x_norm

#-------------------------------------------------------------------------
""" Instance normalisation from StyleGAN """

def instance_norm(x):
    x_mu = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x_sig = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
    x = (x - x_mu) / (x_sig + 1e-8)

    return x

#-------------------------------------------------------------------------
""" Style modulation layer from StyleGAN - maps latent W to
    affine transforms for each generator block """

class StyleModulation(keras.layers.Layer):

    """ nf: number of feature maps in corresponding generator block """

    def __init__(self, nf, name=None):
        super().__init__(name=name)

        self.nf = nf
        self.dense = EqLrDense(units=nf * 2, kernel_initializer=keras.initializers.RandomNormal(0, 1), name="dense")
    
    def call(self, x, w):

        """ x: feature maps from conv stack
            w: latent vector """

        w = self.dense(w, gain=1.0)
        w = tf.reshape(w, [-1, 2, 1, 1, self.nf])

        # Style components
        ys = w[:, 0, :, :, :] + 1 # I.e. initialise bias to 1
        yb = w[:, 1, :, :, :]

        return x * ys + yb

#-------------------------------------------------------------------------
""" Additive noise layer from StyleGAN """

class AdditiveNoise(keras.layers.Layer):

    """ nf: number of feature maps in corresponding generator block """

    def __init__(self, nf, name=None):
        super().__init__(name=name)

        self.nf = nf
        self.noise_weight = self.add_weight(name=f"{self.name}/noise_weight", shape=[1, 1, 1, nf], initializer="zeros", trainable=True)
    
    def call(self, x):
        NHW = x.shape[0:3]
        noise = tf.random.normal(NHW + [1])

        return x + self.noise_weight * noise
