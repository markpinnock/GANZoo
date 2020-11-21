import tensorflow as tf
import tensorflow.keras as keras


""" Inspired by https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/ """


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


# TODO: implement equalised learning rate

# keras.constraints.MaxNorm(1)
# class DiscriminatorBlock(keras.layers.Layer):