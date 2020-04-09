import tensorflow as tf
import tensorflow.keras as keras


def discriminatorModel():
    inputlayer = keras.layers.Input(shape=(512, 512, 1, ))

    dconv1 = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)(inputlayer)
    LR1 = tf.nn.leaky_relu(dconv1, alpha=0.2)
    dconv2 = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)(LR1)
    BN2 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(dconv2), alpha=0.2)
    dconv3 = keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN2)
    BN3 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(dconv3), alpha=0.2)
    dconv4 = keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN3)
    BN4 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(dconv4), alpha=0.2)
    dconv5 = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN4)
    BN5 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(dconv5), alpha=0.2)
    dconv6 = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN5)
    BN6 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(dconv6), alpha=0.2)
    dconv7 = keras.layers.Conv2D(1024, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN6)
    BN7 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(dconv7), alpha=0.2)
    dconv8 = keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='VALID', use_bias=True, activation='linear')(BN7)

    return keras.Model(inputs=inputlayer, outputs=dconv8)


def generatorModel():
    inputlayer = keras.layers.Input(shape=(100, ))
    reshaped = keras.layers.Reshape((1, 1, 100))(inputlayer)

    tconv1 = keras.layers.Conv2DTranspose(1024, (4, 4), strides=(1, 1), padding='VALID', use_bias=False)(reshaped)
    BN1 = tf.nn.relu(keras.layers.BatchNormalization()(tconv1))
    tconv2 = keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN1)
    BN2 = tf.nn.relu(keras.layers.BatchNormalization()(tconv2))
    tconv3 = keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN2)
    BN3 = tf.nn.relu(keras.layers.BatchNormalization()(tconv3))
    tconv4 = keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN3)
    BN4 = tf.nn.relu(keras.layers.BatchNormalization()(tconv4))
    tconv5 = keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN4)
    BN5 = tf.nn.relu(keras.layers.BatchNormalization()(tconv5))
    tconv6 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN5)
    BN6 = tf.nn.relu(keras.layers.BatchNormalization()(tconv6))
    tconv7 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(BN6)
    BN7 = tf.nn.relu(keras.layers.BatchNormalization()(tconv7))
    tconv8 = keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, activation='tanh')(BN7)

    return keras.Model(inputs=inputlayer, outputs=tconv8)
    