import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.eager import context


""" Fade in, minibatch std and pixel norm implementation
    inspired by https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/ """


class EqLrConv2D(keras.layers.Conv2D):

    """ Overloaded implementation of Conv2D layer
        for equalised learning rate, taken from
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py """

    def __init__(self, **kwargs):
        """ Initialise Conv2D with the usual arguments """
        super(EqLrConv2D, self).__init__(**kwargs)
        
        self.weight_scale = None
    
    def call(self, inputs):
        """ Overloaded call to apply weight scale at runtime """
        if self.weight_scale is None:
            fan_in = tf.reduce_prod(self.kernel.shape[:-1])
            self.weight_scale = tf.cast(tf.sqrt(2 / fan_in), tf.float32)

        # Perform convolution and add bias weights
        outputs = self._convolution_op(inputs, self.kernel * self.weight_scale)
        outputs = tf.nn.bias_add(outputs, self.bias, data_format="NHWC")

        # Activation not needed
        return outputs


class EqLrConv2DTranspose(keras.layers.Conv2DTranspose):

    """ Overloaded implementation of Conv2DTranspose layer
        for equalised learning rate - will work only
        for (1, 1, 1) -> (4, 4, N) transpose conv - taken from
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py"""

    def __init__(self, **kwargs):
        """ Initialise Conv2DTranspose with the usual arguments """

        super(EqLrConv2DTranspose, self).__init__(**kwargs)
        
        self.weight_scale = None

    def call(self, inputs):
        """ Overloaded call to apply weight scale at runtime """
        if not self.weight_scale:
            fan_in = tf.reduce_prod(self.kernel.shape[:-1])
            self.weight_scale = tf.cast(tf.sqrt(2 / fan_in), tf.float32)
        
        # 4x4xN output only
        inputs_shape = inputs.shape
        batch_size = inputs_shape[0]
        out_height, out_width = 4, 4
        output_shape = (batch_size, out_height, out_width, self.filters)
        output_shape_tensor = tf.stack(output_shape)
        
        outputs = keras.backend.conv2d_transpose(
            inputs,
            self.kernel * self.weight_scale,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if not context.executing_eagerly():
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs,self.bias, data_format="NHWC")

        # Activation not needed
        return outputs


class FadeInLayer(keras.layers.Layer):
    def __init__(self):
        super(FadeInLayer, self).__init__()

    def call(self, alpha, xs):
        assert (len(xs)) == 2
        return (1.0 - alpha) * xs[0] + alpha * xs[1]
    

class MinibatchStd(keras.layers.Layer):
    def __init__(self):
        super(MinibatchStd, self).__init__()
    
    def call(self, x):
        mean_std_dev = tf.reduce_mean(tf.math.reduce_std(x, axis=0, keepdims=True), keepdims=True)
        stat_channel = tf.tile(mean_std_dev, x.shape[:-1] + [1])
        return tf.concat([x, stat_channel], axis=-1)


class PixelNorm(keras.layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
    
    def call(self, x):
        x_sq = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        x_norm = tf.sqrt(x_sq + 1e-8)
        return x / x_norm


class GANBlock(keras.layers.Layer):
    def __init__(self, nc, initialiser, weight_const, batchnorm, transpose):
        super(GANBlock, self).__init__()
        self.batchnorm = batchnorm
        # TODO: change BN to LN in WGAN-GP
        if transpose:
            self.conv = keras.layers.Conv2DTranspose(nc, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=initialiser, kernel_constraint=weight_const)
        else:
            self.conv = keras.layers.Conv2D(nc, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=initialiser, kernel_constraint=weight_const)
        
        if self.batchnorm:
            self.bn = keras.layers.BatchNormalization()
        
        def call(self, x):
            x = self.conv(x)

            if self.batchnorm:
                x = self.bn(x)
            
            return tf.nn.leaky_relu(x, alpha=0.2)


class ProgGANDiscBlock(keras.layers.Layer):
    def __init__(self, ch, next_block, res, GAN_type, weight_const):
        super(ProgGANDiscBlock, self).__init__()
        double_ch = np.min([ch * 2, res])

        if GAN_type == "progressive":
            Conv2D = EqLrConv2D
            initialiser = keras.initializers.RandomNormal(0, 1)
        else:
            Conv2D = keras.layers.Conv2D 
            initialiser = keras.initializers.RandomNormal(0, 0.02)

        # conv1 only used if first block
        self.next_block = next_block
        self.from_rgb = Conv2D(filters=ch, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
        
        # If this is last discriminator block, collapse to prediction
        if next_block == None:
            self.mb_stats = MinibatchStd()
            self.conv2 = Conv2D(filters=double_ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.out = Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), padding="VALID", kernel_initializer=initialiser, kernel_constraint=weight_const)
        
        # If next blocks exist, conv and downsample
        else:
            self.conv2 = Conv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.conv3 = Conv2D(filters=double_ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.downsample = keras.layers.AveragePooling2D()
            self.fade_in = FadeInLayer()

    def call(self, x, alpha=None, first_block=True):
        
        # If fade in, cache downsampled input image
        if first_block and alpha != None and self.next_block != None:
            next_rgb = self.downsample(x)
            next_rgb = tf.nn.leaky_relu(self.next_block.from_rgb(next_rgb), alpha=0.2)

        # If the very first block, perform 1x1 conv
        if first_block:
            x = tf.nn.leaky_relu(self.from_rgb(x), alpha=0.2)

        # If this is not the last block
        if self.next_block != None:
            x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)
            x = tf.nn.leaky_relu(self.conv3(x), alpha=0.2)
            x = self.downsample(x)

            # If fade in, merge with cached layer
            if first_block and alpha != None and self.next_block != None:
                x = self.fade_in(alpha, [next_rgb, x])
            
            x = self.next_block(x, alpha=None, first_block=False)
        
        # If this is the last block
        else:
            x = self.mb_stats(x)
            x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)
            x = self.out(x)

        return x


class ProgGANGenBlock(keras.layers.Layer):
    def __init__(self, latent_dims, ch, prev_block, GAN_type, weight_const):
        super(ProgGANGenBlock, self).__init__()

        self.prev_block = prev_block
        self.pixel_norm = PixelNorm()

        if GAN_type == "progressive":
            Conv2D = EqLrConv2D
            Conv2DTranspose = EqLrConv2DTranspose
            initialiser = keras.initializers.RandomNormal(0, 1)
        else:
            Conv2D = keras.layers.Conv2D
            Conv2DTranspose = keras.layers.Conv2DTranspose
            initialiser = keras.initializers.RandomNormal(0, 0.02)
        
        # If this is first generator block, reshape latent noise
        if prev_block == None:
            self.reshaped = keras.layers.Reshape((1, 1, latent_dims))
            self.conv1 = Conv2DTranspose(filters=ch, kernel_size=(4, 4), strides=(1, 1), padding="VALID", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.conv2 = Conv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
        
        # If previous blocks exist, we use those
        else:
            self.upsample = keras.layers.UpSampling2D()
            self.conv1 = Conv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.conv2 = Conv2D(filters=ch, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)
            self.fade_in = FadeInLayer()
        
        # Output to rgb
        self.to_rgb = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer=initialiser, kernel_constraint=weight_const)

    def call(self, x, alpha=None, last_block=True):

        # If first block, upsample noise
        if self.prev_block == None:
            x = self.reshaped(x)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv1(x)), alpha=0.2)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv2(x)), alpha=0.2)
        
        # If previous blocks, upsample to_rgb and cache for fade in
        else:
            prev_x, prev_rgb = self.prev_block(x, alpha=None, last_block=False)
            prev_x = self.upsample(prev_x)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv1(prev_x)), alpha=0.2)
            x = tf.nn.leaky_relu(self.pixel_norm(self.conv2(x)), alpha=0.2)

        # Create output image
        rgb = self.to_rgb(x)

        # If fade in, merge cached prev block and this block
        if alpha != None and self.prev_block != None:
            prev_rgb = self.upsample(prev_rgb)
            rgb = self.fade_in(alpha, [prev_rgb, rgb])
        
        if last_block:
            return rgb
        else:
            return x, rgb
