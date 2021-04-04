import tensorflow as tf
import tensorflow.keras as keras


def discriminatorModel():
    inputlayer = keras.layers.Input(shape=(128, 128, 1, ))

    conv1 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)(inputlayer)
    lr1 = tf.nn.leaky_relu(conv1, alpha=0.2)
    conv2 = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(lr1)
    bn2 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(conv2), alpha=0.2)
    conv3 = keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(bn2)
    bn3 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(conv3), alpha=0.2)
    conv4 = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(bn3)
    bn4 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(conv4), alpha=0.2)
    conv5 = keras.layers.Conv2D(1024, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(bn4)
    bn5 = tf.nn.leaky_relu(keras.layers.BatchNormalization()(conv5), alpha=0.2)
    conv6 = keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='VALID', use_bias=True, activation='linear')(bn5)

    return keras.Model(inputs=inputlayer, outputs=conv6)


def generatorModel():
    inputlayer = keras.layers.Input(shape=(256, ))
    reshaped = keras.layers.Reshape((1, 1, 256))(inputlayer)

    tconv1 = keras.layers.Conv2DTranspose(1024, (4, 4), strides=(1, 1), padding='VALID', use_bias=False)(reshaped)
    bn1 = tf.nn.relu(keras.layers.BatchNormalization()(tconv1))
    tconv2 = keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(bn1)
    bn2 = tf.nn.relu(keras.layers.BatchNormalization()(tconv2))
    tconv3 = keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(bn2)
    bn3 = tf.nn.relu(keras.layers.BatchNormalization()(tconv3))
    tconv4 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(bn3)
    bn4 = tf.nn.relu(keras.layers.BatchNormalization()(tconv4))
    tconv5 = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(bn4)
    bn5 = tf.nn.relu(keras.layers.BatchNormalization()(tconv5))
    tconv6 = keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='SAME', use_bias=True, activation='tanh')(bn5)

    return keras.Model(inputs=inputlayer, outputs=tconv6)
    