import abc
import tensorflow as tf

from utils.dataloaders import DiffAug

from utils.losses import (
    minimax_D,
    minimax_G,
    mod_minimax_G,
    least_squares_D,
    least_squares_G,
    wasserstein_D,
    wasserstein_G,
)

class BaseGAN(tf.keras.Model, abc.ABC):

    @abc.abstractmethod
    def __init__(self, config):
        super().__init__(name="GAN")
        self.config = config
        self.latent_dims = config["LATENT_DIM"]
        self.n_critic = config["N_CRITIC"]

        if config["AUGMENT"]:
            self.Aug = DiffAug({"colour": True, "translation": True, "cutout": True})
        else:
            self.Aug = None
        
        self.Generator = None
        self.Discriminator = None

    def compile(self, g_optimiser, d_optimiser, loss):
        super().compile(run_eagerly=False)

        loss_dict = {
            "minmax": [minimax_D, minimax_G],
            "mod_minimax":[minimax_D, mod_minimax_G],
            "least_square": [least_squares_D, least_squares_G],
            "wasserstein": [wasserstein_D, wasserstein_G],
            "wasserstein-GP": [wasserstein_D, wasserstein_G]
            }

        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser

        self.d_loss, self.g_loss = loss_dict[loss]
        self.g_metric = tf.keras.metrics.Mean(name="g_metric")
        self.d_metric = tf.keras.metrics.Mean(name="d_metric")
        self.loss_fn = loss

        self.fixed_noise = tf.random.normal((self.config["NUM_EXAMPLES"], self.latent_dims), dtype=tf.float32)

    def generator_step(self):
        """ Generator training """

        latent_noise = tf.random.normal((self.mb_size, self.latent_dims), dtype=tf.float32)

        with tf.GradientTape() as g_tape:
            g_fake_images = self.Generator(latent_noise, training=True)

            if self.Aug:
                g_fake_images = self.Aug.augment(g_fake_images)

            g_pred = self.Discriminator(g_fake_images, training=True)
            g_loss = self.g_loss(g_pred)
        
        g_grads = g_tape.gradient(g_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))
        self.g_metric.update_state(g_loss)

    def discriminator_step(self, real_images):
        """ Discriminator training """

        mb_size = tf.shape(real_images)[0] // self.n_critic

        # Critic training loop
        for idx in range(self.n_critic):
            # Select minibatch of real images and generate fake images
            d_real_batch = real_images[idx * mb_size:(idx + 1) * mb_size, :, :, :]
            latent_noise = tf.random.normal((self.mb_size, self.latent_dims), dtype=tf.float32)
            d_fake_images = self.Generator(latent_noise, training=True)

            # DiffAug if required
            if self.Aug:
                d_real_batch = self.Aug.augment(d_real_batch)
                d_fake_images = self.Aug.augment(d_fake_images)

            # Get gradients from critic predictions and update weights
            with tf.GradientTape() as d_tape:
                d_pred_fake = self.Discriminator(d_fake_images, training=True)
                d_pred_real = self.Discriminator(d_real_batch, training=True)
                d_loss = self.d_loss(d_pred_real, d_pred_fake)
            
                # Gradient penalty if indicated
                if self.loss_fn == "wasserstein-GP":
                    d_loss += self.Discriminator.apply_WGAN_GP(d_real_batch, d_fake_images)
            
            d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

            # Update metrics
            self.d_metric.update_state(d_loss)

    @abc.abstractmethod
    def train_step(self, data):
        raise NotImplementedError
    
    def summary(self):
        # TODO: modify to work with different scales
        img_dims = [self.config["MAX_RES"], self.config["MAX_RES"], 3]
        latent_dims = [self.config["LATENT_DIM"]]
        inputs = tf.keras.Input(shape=latent_dims)
        outputs = self.Generator.call(inputs)
        print("===========================================================")
        print("Generator")
        print("===========================================================")
        tf.keras.Model(inputs=inputs, outputs=outputs).summary()

        inputs = tf.keras.Input(shape=img_dims)
        outputs = self.Discriminator.call(inputs)
        print("===========================================================")
        print("Discriminator")
        print("===========================================================")
        tf.keras.Model(inputs=inputs, outputs=outputs).summary()        
    
    @abc.abstractmethod
    def call(self):
        raise NotImplementedError
    
    @property
    def metrics(self):
        return [self.d_metric, self.g_metric]
