import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def least_square_loss(labels, predictions):
    if tf.reduce_sum(labels) > 0:
        fake_loss = 0.5 * tf.reduce_mean(tf.pow(predictions[0:labels.shape[0] // 2] - 1, 2))
        real_loss = 0.5 * tf.reduce_mean(tf.pow(predictions[labels.shape[0] // 2:], 2))
        return fake_loss + real_loss
    else:
        return 0.5 * tf.reduce_mean(tf.pow(predictions, 2))


class Discriminator(keras.Model):
    def __init__(self, d_nc):
        super(Discriminator, self).__init__()

        self.conv1 = keras.layers.Conv2D(d_nc, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)
        self.conv2 = keras.layers.Conv2D(d_nc * 2, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)
        self.conv3 = keras.layers.Conv2D(d_nc * 4, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)
        self.conv4 = keras.layers.Conv2D(d_nc * 8, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)
        self.conv5 = keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='VALID', use_bias=True)

        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()

    def call(self, x, training):
        h1 = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)
        h2 = tf.nn.leaky_relu(self.bn2(self.conv2(h1), training=training), alpha=0.2)
        h3 = tf.nn.leaky_relu(self.bn3(self.conv3(h2), training=training), alpha=0.2)
        h4 = tf.nn.leaky_relu(self.bn4(self.conv4(h3), training=training), alpha=0.2)

        return tf.squeeze(self.conv5(h6))


class Generator(keras.Model):
    def __init__(self, noise_dims, g_nc):
        super(Generator, self).__init__()

        self.reshaped = keras.layers.Reshape((1, 1, noise_dims))
        self.tconv1 = keras.layers.Conv2DTranspose(g_nc * 8, (4, 4), strides=(1, 1), padding='VALID', use_bias=True)
        self.tconv2 = keras.layers.Conv2DTranspose(g_nc * 4, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)
        self.tconv3 = keras.layers.Conv2DTranspose(g_nc * 2, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)
        self.tconv4 = keras.layers.Conv2DTranspose(g_nc, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)
        self.tconv5 = keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='SAME', use_bias=True)

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

        return tf.nn.tanh(self.tconv6(h4))


class GAN(keras.Model):
    def __init__(self, noise_dims, g_nc, d_nc, g_optimiser, d_optimiser):
        super(GAN, self).__init__()
        self.noise_dims = noise_dims

        self.loss_dict = {
            "original": keras.losses.BinaryCrossentropy(from_logits=True),
            "least_square": least_square_loss,
            "wasserstein": None
            }

        self.metric_dict = {
            "original": [
                keras.metrics.BinaryCrossentropy(from_logits=True),
                keras.metrics.BinaryCrossentropy(from_logits=True)
                ],
            "least_square": [
                keras.metrics.Mean(),
                keras.metrics.Mean()
                ],
            "wasserstein": None
        }

        self.loss = self.loss_dict["original"]
        self.g_metric = self.metric_dict["original"][0]
        self.d_metric = self.metric_dict["original"][1]
        self.Generator = Generator(noise_dims, g_nc)
        self.Discriminator = Discriminator(d_nc)
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
    
    def compile(self, g_optimiser, d_optimiser, loss_key):
        super(GAN, self).compile()
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.loss = self.loss_dict[loss_key]
    
    def train_step(self, real_images):
        mb_size = real_images.shape[0]
        noise = tf.random.normal((mb_size, self.noise_dims), dtype=tf.float32)
        d_fake_images = self.Generator(noise, training=True)
        d_labels = tf.concat([tf.ones((mb_size, 1)), tf.zeros((mb_size, 1))], axis=0)

        # ADD NOISE TO LABELS AND/OR IMAGES

        with tf.GradientTape() as d_tape:
            d_pred_fake = self.Discriminator(d_fake_images, training=True)
            d_pred_real = self.Discriminator(real_images, training=True)
            d_predictions = tf.concat([d_pred_fake, d_pred_real], axis=0)
            d_loss = self.loss(d_labels, d_predictions)
        
        d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
        self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))
        self.d_metric.update_state(d_labels, d_predictions)
        # self.d_metric.update_state(d_loss)

        noise = tf.random.normal((mb_size, self.noise_dims), dtype=tf.float32)
        g_labels = tf.zeros((mb_size, 1))
        # ADD NOISE TO LABELS AND/OR IMAGES

        with tf.GradientTape() as g_tape:
            g_fake_images = self.Generator(noise, training=True)
            g_predictions = self.Discriminator(g_fake_images, training=True)
            g_loss = self.loss(g_labels, g_predictions)
        
        g_grads = g_tape.gradient(g_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))
        self.g_metric.update_state(g_labels, g_predictions)
        # self.g_metric.update_state(g_loss)

        return {"g_loss": g_loss, "d_loss": d_loss}
