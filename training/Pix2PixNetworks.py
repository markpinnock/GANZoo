import tensorflow as tf
import tensorflow.keras as keras


def dnBlock(nc, inputlayer, strides, batchnorm=True):
    conv = keras.layers.Conv3D(nc, (4, 4, 2), strides=strides, padding='same')(inputlayer)

    if batchnorm:
        conv = keras.layers.BatchNormalization()(conv)
    
    out = tf.nn.leaky_relu(conv, alpha=0.2)

    return out


def upBlock(nc, inputlayer, skip, tconv_strides, dropout=False):
    tconv = keras.layers.Conv3DTranspose(nc, (4, 4, 2), strides=tconv_strides, padding='same')(inputlayer)
    BN = keras.layers.BatchNormalization()(tconv)

    if dropout:
        BN = keras.layers.Dropout(0.5)(BN)
    
    out = tf.nn.relu(BN)
    concat = tf.nn.relu(keras.layers.concatenate([out, skip], axis=4))

    return concat


def discriminatorModel(image_dims):
    fake_img = keras.layers.Input(shape=image_dims)
    real_img = keras.layers.Input(shape=image_dims)
    concat = keras.layers.concatenate([fake_img, real_img], axis=4)
    # in_img = keras.layers.Input(shape=image_dims)

    dconv1 = dnBlock(64, concat, (2, 2, 2), batchnorm=False) # 128 128 6
    dconv2 = dnBlock(128, dconv1, (2, 2, 1)) # 64 64 6
    dconv3 = dnBlock(256, dconv2, (2, 2, 2)) # 32 32 3
    dconv4 = dnBlock(512, dconv3, (2, 2, 1)) # 16 16 3
    dconv5 = dnBlock(512, dconv4, (2, 2, 2)) # 8 8 1
    dconv6 = dnBlock(512, dconv5, (2, 2, 2)) # 4 4 1
    dconv7 = keras.layers.Conv3D(1, (4, 4, 1), strides=(1, 1, 1), padding='VALID', activation='linear')(dconv6)

    return keras.Model(inputs=[fake_img, real_img], outputs=dconv7)


def generatorModel(input_dims):
    lo_img = keras.layers.Input(shape=input_dims)

    dn1 = dnBlock(64, lo_img, (2, 2, 2), batchnorm=False) # 256 256 6
    dn2 = dnBlock(128, dn1, (2, 2, 1)) 
    dn3 = dnBlock(256, dn2, (2, 2, 2)) # 64 64 3
    dn4 = dnBlock(512, dn3, (2, 2, 1))
    dn5 = dnBlock(512, dn4, (2, 2, 1)) # 16 16 3

    dn6 = dnBlock(512, dn5, (2, 2, 1), batchnorm=False) # 8 8 3

    up5 = upBlock(512, dn6, dn5, (2, 2, 1), dropout=True)
    up4 = upBlock(512, up5, dn4, (2, 2, 1), dropout=True)
    up3 = upBlock(256, up4, dn3, (2, 2, 1), dropout=True) 
    up2 = upBlock(128, up3, dn2, (2, 2, 2))
    up1 = upBlock(64, up2, dn1, (2, 2, 1))

    outputlayer = keras.layers.Conv3DTranspose(1, (4, 4, 4), strides=(2, 2, 2), padding='same', activation='tanh')(up1)

    return keras.Model(inputs=lo_img, outputs=outputlayer)