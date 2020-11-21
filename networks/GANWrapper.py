import tensorflow as tf
import tensorflow.keras as keras

from networks.Networks import Discriminator, Generator
from utils.TrainFuncs import least_square_loss, wasserstein_loss, gradient_penalty


class GAN(keras.Model):

    """ GAN class
        - latent_dims: size of generator latent distribution
        - g_nc: number of channels in generator first layer
        - d_nc: number of channels in discriminator first layer
        - g_optimiser: generator optimiser e.g. keras.optimizers.Adam()
        - d_optimiser: discriminator optimiser e.g. keras.optimizers.Adam()
        - GAN_type: 'original', 'least_square', 'wasserstein' or 'wasserstein-GP'
        - n_critic: number of discriminator/critic training runs (5 in WGAN, 1 otherwise) """

    def __init__(self, latent_dims, g_nc, d_nc, g_optimiser, d_optimiser, GAN_type, n_critic):
        super(GAN, self).__init__()
        self.GAN_type = GAN_type
        self.latent_dims = latent_dims
        self.initialiser = keras.initializers.RandomNormal(0, 0.02)

        # Choose appropriate loss and initialise metrics
        self.loss_dict = {
            "original": keras.losses.BinaryCrossentropy(from_logits=True),
            "least_square": least_square_loss,
            "wasserstein": wasserstein_loss,
            "wasserstein-GP": wasserstein_loss,
            "progressive": wasserstein_loss
            }

        self.metric_dict = {
            "g_metric": keras.metrics.Mean(),
            "d_metric_1": keras.metrics.Mean(),
            "d_metric_2": keras.metrics.Mean()
        }

        # Set up real/fake labels
        if GAN_type == "wasserstein":
            self.d_real_label = -1.0
            self.d_fake_label = 1.0
            self.g_label = -1.0
            cons = True
        elif GAN_type == "wasserstein-GP":
            self.d_real_label = -1.0
            self.d_fake_label = 1.0
            self.g_label = -1.0
            cons = False
        elif GAN_type == "progressive":
            self.d_real_label = -1.0
            self.d_fake_label = 1.0
            self.g_label = -1.0
            cons = "maxnorm"
        else:
            self.d_real_label = 0.0
            self.d_fake_label = 1.0
            self.g_label = 0.0
            cons = False
        # TODO: IMPLEMENT CONSTRAINT TYPE
        self.loss = self.loss_dict[GAN_type]
        self.Generator = Generator(latent_dims, g_nc, self.initialiser, cons)
        self.Discriminator = Discriminator(d_nc, self.initialiser, cons)
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.n_critic = n_critic
        self.fade_iter = 0
        self.fade_count = 0
        self.alpha = 0
    
    def compile(self, g_optimiser, d_optimiser, loss_key):
        # Not currently used
        raise NotImplementedError
        super(GAN, self).compile()
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.loss = self.loss_dict[loss_key]
    
    def fade_set(self, num_iter):
        """ Activates or deactivates fade in """

        self.fade_iter = num_iter
        self.fade_count = 0

    def set_trainable_layers(self, scale):
        """ Sets new block to trainable and sets to_rgb/from_rgb
            conv layers in old blocks to untrainable
            to avoid missing gradients """ 

        self.Discriminator.blocks[scale].trainable = True
        
        for i in range(0, scale):
            self.Discriminator.blocks[i].from_rgb.trainable = False
        
        self.Generator.blocks[scale].trainable = True
        
        for i in range(0, scale):
            self.Generator.blocks[i].to_rgb.trainable = False

    # @tf.function
    def train_step(self, real_images, scale):
        # Determine labels and size of mb for each critic training run
        # (size of real_images = minibatch size * number of critic runs)
        mb_size = real_images.shape[0] // self.n_critic

        d_labels = tf.concat(
            [tf.ones((mb_size, 1)) * self.d_fake_label,
             tf.ones((mb_size, 1)) * self.d_real_label
             ], axis=0)
            
        g_labels = tf.ones((mb_size, 1)) * self.g_label

        if self.fade_iter:
            self.Discriminator.alpha = self.fade_count / self.fade_iter
            self.Generator.alpha = self.fade_count / self.fade_iter
        else:
            self.Discriminator.alpha = None
            self.Generator.alpha = None
        # TODO: ADD NOISE TO LABELS AND/OR IMAGES

        # Critic training loop
        for idx in range(self.n_critic):
            # Select minibatch of real images and generate fake images
            d_real_batch = real_images[idx * mb_size:(idx + 1) * mb_size, :, :, :]
            latent_noise = tf.random.normal((mb_size, self.latent_dims), dtype=tf.float32)
            d_fake_images = self.Generator(latent_noise, scale, training=True)

            # Get gradients from critic predictions and update weights
            with tf.GradientTape() as d_tape:
                d_pred_fake = self.Discriminator(d_fake_images, scale, training=True)
                d_pred_real = self.Discriminator(d_real_batch, scale, training=True)
                d_predictions = tf.concat([d_pred_fake, d_pred_real], axis=0)
                d_loss_1 = self.loss(d_labels[0:mb_size], d_predictions[0:mb_size]) # Fake
                d_loss_2 = self.loss(d_labels[mb_size:], d_predictions[mb_size:]) # Real
                d_loss = 0.5 * d_loss_1 + 0.5 * d_loss_2
            
                # Gradient penalty if indicated
                # TODO: tidy up loss selection
                if self.GAN_type == "wasserstein-GP" or "progressive":
                    grad_penalty = gradient_penalty(d_real_batch, d_fake_images, self.Discriminator, scale)
                    d_loss += 10 * grad_penalty
            
            d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

            # Update metrics
            self.metric_dict["d_metric_1"].update_state(d_loss_1)
            self.metric_dict["d_metric_2"].update_state(d_loss_2)

        # Generator training
        noise = tf.random.normal((mb_size, self.latent_dims), dtype=tf.float32)
        
        # TODO: ADD NOISE TO LABELS AND/OR IMAGES

        # Get gradients from critic predictions of generated fake images and update weights
        with tf.GradientTape() as g_tape:
            g_fake_images = self.Generator(noise, scale, training=True)
            g_predictions = self.Discriminator(g_fake_images, scale, training=True)
            g_loss = self.loss(g_labels, g_predictions)
        
        g_grads = g_tape.gradient(g_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))

        # Update metric and increment fade count
        self.metric_dict["g_metric"].update_state(g_loss)
        self.fade_count += 1
