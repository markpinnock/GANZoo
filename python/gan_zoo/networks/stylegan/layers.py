import numpy as np
import tensorflow as tf


#-------------------------------------------------------------------------
""" Overloaded implementation of Dense layer for equalised learning rate
    by initialising with N(0, 1) and scale weights at run-time instead of
    using He initialiser - prevents escalating signal magnitudes """

class EqLrDense(tf.keras.layers.Dense):

    def __init__(self, gain=tf.sqrt(2.0), lr_mul=1.0, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.lr_mul = lr_mul
    
    def build(self, input_shape):
        # Call layer's native build method to initialise weights
        super().build(input_shape)
        fan_in = tf.shape(self.kernel)[0]

        # Std dev multiplier from He initialiser
        self.wscale = self.gain / tf.sqrt(tf.cast(fan_in, "float32"))

    def call(self, x):
        # Scale matrix before matrix multiplication and bias
        x = tf.matmul(x, self.kernel * self.wscale * self.lr_mul)
        x = tf.add(x, self.bias * self.lr_mul)

        return x


#-------------------------------------------------------------------------
""" Overloaded implementation of Conv2D layer for equalised learning rate
    by initialising with N(0, 1) and scale weights at run-time instead of
    using He initialiser - prevents escalating signal magnitudes """

class EqLrConv2D(tf.keras.layers.Conv2D):

    def __init__(self, gain=tf.sqrt(2.0), **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
    
    def build(self, input_shape):
        # Call layer's native build method to initialise weights
        super().build(input_shape)
        fan_in = tf.reduce_prod(tf.shape(self.kernel)[:-1])

        # Std dev multiplier from He initialiser
        self.wscale = self.gain / tf.sqrt(tf.cast(fan_in, "float32"))
        
    def call(self, x):
        # Scale conv kernel before convolution and bias
        x = self._convolution_op(x, self.kernel * self.wscale)
        x = tf.add(x, self.bias)

        return x


#-------------------------------------------------------------------------
""" Fade in - interpolates between two layers by factor alpha """

def fade_in(alpha, old, new):
    with tf.name_scope("fade_in") as scope:
        return (1.0 - alpha) * old + alpha * new


#-------------------------------------------------------------------------
""" Minibatch standard deviation from ProgGan and StyleGAN - increases
    variation by appending feature statistics to discriminator """

def mb_stddev(x, group_size=4, ch_size=1):
    with tf.name_scope("mb_stddev") as scope:
        dims = tf.shape(x)
        group_size = tf.reduce_min([group_size, dims[0]])

        # Split minibatches into G groups of M and channels into N groups of C
        y = tf.reshape(x, [group_size, -1, dims[1], dims[2], ch_size, dims[3] // ch_size]) # [G, M, H, W, N, C]

        # Calculate std dev of each feature, then avg over group
        y = tf.math.reduce_std(y, axis=0)                                                  # [M, H, W, N, C]
        y = tf.reduce_mean(y, axis=[1, 2, 4], keepdims=True)                               # [M, 1, 1, N, 1]
        y = tf.reduce_mean(y, axis=[4])                                                    # [M, 1, 1, N]

        # Expand and append to features
        y = tf.tile(y, [group_size, dims[1], dims[2], 1])                                  # [N, H, W, 1]

        return tf.concat([x, y], axis=-1, name=scope)


#-------------------------------------------------------------------------
""" Pixel normalisation from StyleGAN mapping network - normalises each
    feature vector to prevent escalation of signal magnitudes """

def pixel_norm(x):
    with tf.name_scope("pixel_norm") as scope:
        x_sq = tf.reduce_mean(tf.square(x, name=f"{scope}_square"), axis=-1, keepdims=True, name=f"{scope}_mean")
        x_norm = tf.sqrt(x_sq + 1e-8, name=f"{scope}_sqrt")
    
        return x / x_norm


#-------------------------------------------------------------------------
""" (Adaptive) Instance normalisation from StyleGAN, scaled and biased by
    style components from intermediate latent space """

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
        ys = w[:, 0, :, :, :] + 1 # I.e. initialise scale to 1
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
