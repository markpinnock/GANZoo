import tensorflow as tf
import tensorflow.keras as keras


#-------------------------------------------------------------------------
""" Original minimax loss
    Goodfellow et al. Generative adversarial networks. NeurIPS, 2014
    https://arxiv.org/abs/1406.2661 """

@tf.function
def minimax_D(real_output, fake_output):
    real_loss = keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
    fake_loss = keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
    return real_loss + fake_loss

@tf.function
def minimax_G(fake_output):
    fake_loss = -keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
    return fake_loss

@tf.function
def mod_minimax_G(fake_output):
    fake_loss = keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)
    return fake_loss

#-------------------------------------------------------------------------
""" Least squares loss
    Mao et al. Least squares generative adversarial networks
    Proceedings of the IEEE International Conference on Computer Vision, 2017
    https://arxiv.org/abs/1611.04076 """

@tf.function
def least_squares_D(real_output, fake_output):
    real_loss = 0.5 * tf.reduce_mean(tf.square(real_img - 1))
    fake_loss = 0.5 * tf.reduce_mean(tf.square(fake_img))
    return fake_loss + real_loss

@tf.function
def least_squares_G(fake_output):
    fake_loss = 0.5 * tf.reduce_mean(tf.square(fake_output - 1))
    return fake_loss

#-------------------------------------------------------------------------
""" Wasserstein loss
    Arjovsky et al. Wasserstein generative adversarial networks.
    International conference on machine learning. PMLR, 2017
    https://arxiv.org/abs/1701.07875 """

@tf.function
def wasserstein_D(real_output, fake_output):
    return tf.reduce_mean(fake_output - real_output)

@tf.function
def wasserstein_G(fake_output):
    return tf.reduce_mean(-fake_output)

class WeightClipConstraint(keras.constraints.Constraint):

    """ Clips weights in WGAN
        - clip_val: value to be clipped to (+/- clip-val) """

    def __init__(self, clip_val):
        self.clip_val = clip_val
    
    def call(self, weights):
        return keras.backend.clip(weights, -self.clip_val, self.clip_val)
    
    def get_config(self):
        return {"clip_value": self.clip_val}

#-------------------------------------------------------------------------
""" Wasserstein loss gradient penalty
    Gulrajani et al. Improved training of Wasserstein GANs. NeurIPS, 2017
    https://arxiv.org/abs/1704.00028 """

@tf.function
def gradient_penalty(real_img, fake_img, D, scale):
    epsilon = tf.random.uniform([fake_img.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = (epsilon * real_img) + ((1 - epsilon) * fake_img)

    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        D_hat = D(x_hat, scale, training=True)
    
    gradients = tape.gradient(D_hat, x_hat)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=(1, 2)) + 1e-8)
    grad_penalty = tf.reduce_mean(tf.square(grad_norm - 1))

    return grad_penalty

