import tensorflow as tf
import tensorflow.keras as keras

from utils.TrainFuncs import WeightClipConstraint


class Discriminator(keras.Model):

    """ Discriminator model for GAN
        - d_nc: number of channels in first layer
        - initaliser: e.g. keras.initalizers.RandomNormal()
        - clip: True/False, whether to clip layer weights """

    def __init__(self, d_nc, initialiser, clip):
        super(Discriminator, self).__init__()
        self.initialiser = initialiser

        if clip:
            self.weight_clip = WeightClipConstraint(0.01)
        else:
            self.weight_clip = None

        self.conv1 = keras.layers.Conv2D(d_nc, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=self.initialiser, kernel_constraint=self.weight_clip)
        self.conv2 = keras.layers.Conv2D(d_nc * 2, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=self.initialiser, kernel_constraint=self.weight_clip)
        self.conv3 = keras.layers.Conv2D(d_nc * 4, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=self.initialiser, kernel_constraint=self.weight_clip)
        self.conv4 = keras.layers.Conv2D(d_nc * 8, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=self.initialiser, kernel_constraint=self.weight_clip)
        self.conv5 = keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='VALID', use_bias=True, kernel_initializer=self.initialiser, kernel_constraint=self.weight_clip)
        # TODO: change BN to LN in WGAN-GP
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()

    def call(self, x, training):
        h1 = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)
        h2 = tf.nn.leaky_relu(self.bn2(self.conv2(h1), training=training), alpha=0.2)
        h3 = tf.nn.leaky_relu(self.bn3(self.conv3(h2), training=training), alpha=0.2)
        h4 = tf.nn.leaky_relu(self.bn4(self.conv4(h3), training=training), alpha=0.2)

        return tf.squeeze(self.conv5(h4))


class Generator(keras.Model):

    """ Generator model for GAN
        - latient_dims: size of latent distribution
        - g_nc: number of channels in first layer
        - initaliser: e.g. keras.initalizers.RandomNormal() """

    def __init__(self, latent_dims, g_nc, initialiser):
        super(Generator, self).__init__()
        self.initialiser = initialiser

        self.reshaped = keras.layers.Reshape((1, 1, latent_dims))
        self.tconv1 = keras.layers.Conv2DTranspose(g_nc * 8, (4, 4), strides=(1, 1), padding='VALID', use_bias=True, kernel_initializer=self.initialiser)
        self.tconv2 = keras.layers.Conv2DTranspose(g_nc * 4, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=self.initialiser)
        self.tconv3 = keras.layers.Conv2DTranspose(g_nc * 2, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=self.initialiser)
        self.tconv4 = keras.layers.Conv2DTranspose(g_nc, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=self.initialiser)
        self.tconv5 = keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, kernel_initializer=self.initialiser)

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()

    def call(self, x, training):
        hr = self.reshaped(x)
        h1 = tf.nn.relu(self.bn1(self.tconv1(hr), training=training))
        h2 = tf.nn.relu(self.bn2(self.tconv2(h1), training=training))
        h3 = tf.nn.relu(self.bn3(self.tconv3(h2), training=training))
        h4 = tf.nn.relu(self.bn4(self.tconv4(h3), training=training))

        return tf.nn.tanh(self.tconv5(h4))

