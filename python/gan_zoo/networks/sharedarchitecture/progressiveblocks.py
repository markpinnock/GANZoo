import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops, nn_ops


#-------------------------------------------------------------------------
""" Progressive and StyleGAN Discriminator block """
# TODO: separate into last and prev blocks
class ProgressiveDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, ch, next_block, config, name=None):
        super().__init__(name=name)
        double_ch = np.min([ch * 2, config["MAX_CHANNELS"]])
        initialiser = tf.keras.initializers.RandomNormal(0, 1)

        self.next_block = next_block
        self.from_rgb = EqLrConv2D(filters=ch, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="from_rgb")

        # If this is last discriminator block, collapse to prediction
        if next_block == None:
            self.conv = EqLrConv2D(filters=double_ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv")
            self.flat = tf.keras.layers.Flatten(name="flatten")
            self.dense = EqLrDense(units=config["LATENT_DIM"], kernel_initializer=initialiser, name="dense")
            self.out = EqLrDense(units=1, kernel_initializer=initialiser, name="out")
        
        # If next blocks exist, conv and downsample
        else:
            self.conv1 = EqLrConv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv1")
            self.conv2 = EqLrConv2D(filters=double_ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, name="conv2")
            self.downsample = tf.keras.layers.AveragePooling2D(name="down2D")

    def call(self, x, fade_alpha=None, first_block=True):

        # If fade in, pass downsampled image into next block and cache
        if first_block and fade_alpha != None and self.next_block != None:
            next_rgb = self.downsample(x)
            next_rgb = tf.nn.leaky_relu(self.next_block.from_rgb(next_rgb), alpha=0.2)

        # If the very first block, perform 1x1 conv on rgb
        if first_block:
            x = tf.nn.leaky_relu(self.from_rgb(x), alpha=0.2)

        # If this is not the last block
        if self.next_block != None:
            x = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)
            x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)
            x = self.downsample(x)

            # If fade in, merge with cached layer
            if first_block and fade_alpha != None and self.next_block != None:
                x = fade_in(fade_alpha, next_rgb, x)
            
            x = self.next_block(x, fade_alpha=None, first_block=False)
        
        # If this is the last block
        else:
            x = mb_stddev(x)
            x = tf.nn.leaky_relu(self.conv(x), alpha=0.2)
            x = self.flat(x)
            x = tf.nn.leaky_relu(self.dense(x))
            x = self.out(x, noise=None, gain=1) # Gain as in original implementation

        return x


#-------------------------------------------------------------------------
""" Overloaded implementation of Dense layer for equalised learning rate,
    taken from https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/core.py """

class EqLrDense(tf.keras.layers.Dense):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight_scale = None
    
    def call(self, inputs, noise=None, gain=tf.sqrt(2.0), lr_mul=1.0):
        """ Overloaded call to apply weight scale at runtime """

        if self.weight_scale is None:
            fan_in = tf.reduce_prod(tf.shape(self.kernel)[:-1])
            self.weight_scale = gain / tf.sqrt(tf.cast(fan_in, tf.float32))

        # Perform dense layer matmul (optional noise step for StyleGAN)
        outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel * self.weight_scale * lr_mul)
        if noise: outputs = noise(outputs)
        outputs = nn_ops.bias_add(outputs, self.bias * lr_mul)
        """Bias scaled??? Check lr factor """
        # Activation not needed
        return outputs


#-------------------------------------------------------------------------
""" Overloaded implementation of Conv2D layer for equalised learning rate,
    taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py """

class EqLrConv2D(tf.keras.layers.Conv2D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight_scale = None
        
    def call(self, inputs, gain=tf.sqrt(2.0)):
        """ Overloaded call method applies weight scale at runtime """

        if self.weight_scale is None: # TODO: implement in .build()
            fan_in = tf.reduce_prod(tf.shape(self.kernel)[:-1])
            self.weight_scale = gain / tf.sqrt(tf.cast(fan_in, tf.float32))

        # Perform convolution and add bias weights (optional noise step for StyleGAN)
        outputs = self._convolution_op(inputs, self.kernel * self.weight_scale)
        outputs = tf.nn.bias_add(outputs, self.bias, data_format="NHWC")

        # Activation not needed
        return outputs


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
        x_sq = tf.reduce_mean(tf.square(x, name="square"), axis=-1, keepdims=True, name="mean")
        x_norm = tf.sqrt(x_sq + 1e-8, name="sqrt")
    
        return x / x_norm
