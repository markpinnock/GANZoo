import tensorflow as tf
import tensorflow.keras as keras
from abc import abstractclassmethod

from networks.Networks import ProStyleGANDiscriminator, ProGANGenerator, StyleGANGenerator
from utils.DataLoaders import DiffAug

from utils.Losses import (
    minimax_D,
    minimax_G,
    mod_minimax_G,
    least_squares_D,
    least_squares_G,
    wasserstein_D,
    wasserstein_G,
    gradient_penalty
)


class BaseGAN(keras.Model):

    def __init__(self, config):
        super().__init__(name="GAN")
        self.latent_dims = config["LATENT_DIM"]
        self.GAN_type = config["MODEL"]
        self.n_critic = config["N_CRITIC"]
        self.loss_fn = config["LOSS_FN"]

        loss_dict = {
            "minimax": [minimax_D, minimax_G],
            "mod_minimax":[minimax_D, minimax_G],
            "least_square": [least_squares_D, least_squares_G],
            "wasserstein": [wasserstein_D, wasserstein_G],
            "wasserstein-GP": [wasserstein_D, wasserstein_G]
            }

        self.metric_dict = {
            "g_metric": keras.metrics.Mean(),
            "d_metric": keras.metrics.Mean()
        }

        opt_dict = {
            "Adam": keras.optimizers.Adam,
            "RMSprop": keras.optimizers.RMSprop
        }

        self.loss_D = loss_dict[self.loss_fn][0]
        self.loss_G = loss_dict[self.loss_fn][1]
        self.g_optimiser = opt_dict[config["G_OPT"]](*config["G_ETA"])
        self.d_optimiser = opt_dict[config["D_OPT"]](*config["D_ETA"])

        if config["AUGMENT"]:
            self.Aug = DiffAug({"colour": True, "translation": True, "cutout": True})
        else:
            self.Aug = None
    
    @abstractclassmethod
    def discriminator_step(self):
        raise NotImplementedError

    @abstractclassmethod
    def generator_step(self):
        raise NotImplementedError

    @abstractclassmethod
    def train_step(self):
        raise NotImplementedError


class DCGAN(BaseGAN):

    def __init__(self, config):
        super().__init__(config)
        self.Generator = Generator(config=config, name="Generator")
        self.Discriminator = Discriminator(config=config, name="Discriminator")

    def discriminator_step(self, real_images):
        mb_size = real_images.shape[0] // self.n_critic

        # Critic training loop
        for idx in range(self.n_critic):
            # Select minibatch of real images and generate fake images
            d_real_batch = real_images[idx * mb_size:(idx + 1) * mb_size, :, :, :]
            latent_noise = tf.random.normal((mb_size, self.latent_dims), dtype=tf.float32)
            d_fake_images = self.Generator(latent_noise, scale, training=True)

            # DiffAug if required
            if self.Aug:
                d_real_batch = self.Aug.augment(d_real_batch)
                d_fake_images = self.Aug.augment(d_fake_images)

            # Get gradients from critic predictions and update weights
            with tf.GradientTape() as d_tape:
                d_pred_fake = self.Discriminator(d_fake_images, training=True)
                d_pred_real = self.Discriminator(d_real_batch, training=True)
                d_loss = self.loss_D(d_pred_real, d_pred_fake)
            
                # Gradient penalty if indicated
                if self.loss_fn == "wasserstein-GP":
                    grad_penalty = gradient_penalty(d_real_batch, d_fake_images, self.Discriminator, scale)
                    d_loss += 10 * grad_penalty
            
            d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

            # Update metrics
            self.metric_dict["d_metric"].update_state(d_loss)

    def generator_step(self):
        # Generator training
        noise = tf.random.normal((mb_size, self.latent_dims), dtype=tf.float32)

        # Get gradients from critic predictions of generated fake images and update weights
        with tf.GradientTape() as g_tape:
            g_fake_images = self.Generator(noise, training=True)
            if self.Aug: g_fake_images = self.Aug.augment(g_fake_images)
            g_pred = self.Discriminator(g_fake_images, training=True)
            g_loss = self.loss_G(g_pred)
        
        g_grads = g_tape.gradient(g_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))
        self.metric_dict["g_metric"].update_state(g_loss)
    
    @tf.function
    def train_step(self, real_images):
        self.discriminator_step(real_images)
        self.generator_step()


class ProStyleGAN(BaseGAN):

    def __init__(self, config):
        super().__init__(config)

        self.Generator = StyleGANGenerator(config=config, name="Generator")
        self.Discriminator = ProStyleGANDiscriminator(config=config, name="Discriminator")

        # Exponential moving average of generator weights for images
        self.EMAGenerator = StyleGANGenerator(config=config, name="EMAGenerator")
        self.update_mvag_generator(initial=True)
        self.EMA_beta = config["EMA_BETA"]
        self.fade_iter = 0
        self.fade_count = 0
        self.alpha = 0
    
    def fade_set(self, num_iter):
        """ Activates or deactivates fade in """

        self.fade_iter = num_iter
        self.fade_count = 0

    def set_trainable_layers(self, scale):
        """ Sets new block to trainable and sets to_rgb/from_rgb
            conv layers in old blocks to untrainable
            to avoid missing gradients warning """ 

        self.Discriminator.blocks[scale].trainable = True
       
        for i in range(0, scale):
            self.Discriminator.blocks[i].from_rgb.trainable = False
        
        self.Generator.blocks[scale].trainable = True
        
        for i in range(0, scale):
            self.Generator.blocks[i].to_rgb.trainable = False

        self.EMAGenerator.blocks[scale].trainable = True
        
        for i in range(0, scale):
            self.EMAGenerator.blocks[i].to_rgb.trainable = False

    def update_mvag_generator(self, initial=False):

        """ Updates EMAGenerator with Generator weights """

        # If first use, clone Generator
        if initial:
            assert len(self.Generator.weights) == len(self.EMAGenerator.weights)

            for idx in range(len(self.EMAGenerator.weights)):
                assert self.EMAGenerator.weights[idx].name == self.Generator.weights[idx].name
                self.EMAGenerator.weights[idx].assign(self.Generator.weights[idx])
            
        else:
            for idx in range(len(self.EMAGenerator.trainable_weights)):
                new_weights = self.EMA_beta * self.EMAGenerator.trainable_weights[idx] + (1 - self.EMA_beta) * self.Generator.trainable_weights[idx]
                self.EMAGenerator.trainable_weights[idx].assign(new_weights)

    def discriminator_step(self, real_images, scale):
        # Determine labels and size of mb for each critic training run
        # (size of real_images = minibatch size * number of critic runs)
        mb_size = real_images.shape[0] // self.n_critic

        # Critic training loop
        for idx in range(self.n_critic):
            # Select minibatch of real images and generate fake images
            d_real_batch = real_images[idx * mb_size:(idx + 1) * mb_size, :, :, :]
            latent_noise = tf.random.normal((mb_size, self.latent_dims), dtype=tf.float32)
            d_fake_images = self.Generator(latent_noise, scale, training=True)

            # DiffAug if required
            if self.Aug:
                d_real_batch = self.Aug.augment(d_real_batch)
                d_fake_images = self.Aug.augment(d_fake_images)

            # Get gradients from critic predictions and update weights
            with tf.GradientTape() as d_tape:
                d_pred_fake = self.Discriminator(d_fake_images, scale, training=True)
                d_pred_real = self.Discriminator(d_real_batch, scale, training=True)
                d_loss = self.loss_D(d_pred_real, d_pred_fake)

                if self.loss_fn == "wasserstein-GP" and self.GAN_type == "ProGAN":
                    d_loss += 0.001 * tf.reduce_mean(tf.square(d_pred_real))

                # Gradient penalty if indicated
                if self.loss_fn == "wasserstein-GP":
                    grad_penalty = gradient_penalty(d_real_batch, d_fake_images, self.Discriminator, scale)
                    d_loss += 10 * grad_penalty
            
            d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

            # Update metrics
            self.metric_dict["d_metric"].update_state(d_loss)

    def generator_step(self, mb_size, scale):
        # Generator training
        noise = tf.random.normal((mb_size, self.latent_dims), dtype=tf.float32)

        # Get gradients from critic predictions of generated fake images and update weights
        with tf.GradientTape() as g_tape:
            g_fake_images = self.Generator(noise, scale, training=True)
            if self.Aug: g_fake_images = self.Aug.augment(g_fake_images)
            g_pred = self.Discriminator(g_fake_images, scale, training=True)
            g_loss = self.loss_G(g_pred)
        
        g_grads = g_tape.gradient(g_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))
        self.metric_dict["g_metric"].update_state(g_loss)
    
    # @tf.function
    def train_step(self, real_images, scale):

        if self.fade_iter:
            self.Discriminator.alpha = self.fade_count / self.fade_iter
            self.Generator.alpha = self.fade_count / self.fade_iter

        else:
            self.Discriminator.alpha = None
            self.Generator.alpha = None

        self.discriminator_step(real_images, scale)
        self.generator_step(real_images.shape[0], scale)

        # Update MVAG and fade count
        self.update_mvag_generator()
        self.fade_count += 1
