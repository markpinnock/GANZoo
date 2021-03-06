import tensorflow as tf
import tensorflow.keras as keras

from networks.Networks import Discriminator, Generator
from utils.DataLoaders import DiffAug

from utils.Losses import (
    minimax_D,
    minimax_G,
    mod_minimax_G,
    least_squares_D,
    wasserstein_D,
    wasserstein_G,
    gradient_penalty
)


class GAN(keras.Model):

    """ GAN class
        - resolution: size of output images
        - g_nc: number of channels in generator last layer
        - d_nc: number of channels in discriminator first layer
        - g_optimiser: generator optimiser e.g. keras.optimizers.Adam()
        - d_optimiser: discriminator optimiser e.g. keras.optimizers.Adam()
        - GAN_type: 'original', 'least_square', 'wasserstein' or 'wasserstein-GP'
        - n_critic: number of discriminator/critic training runs (5 in WGAN, 1 otherwise) """

    def __init__(self, config, g_optimiser, d_optimiser, n_critic):
        super().__init__()
        self.latent_dims = config["LATENT_DIM"]
        self.GAN_type = config["MODEL"]

        # Choose appropriate loss and initialise metrics
        self.loss_dict = {
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

        self.loss_D = self.loss_dict[config["LOSS_FN"]][0]
        self.loss_G = self.loss_dict[config["LOSS_FN"]][1]

        self.Generator = Generator(config=config, name="Generator")
        self.Discriminator = Discriminator(config=config, name="Discriminator")

        # Exponential moving average of generator weights for images
        self.EMAGenerator = Generator(config=config, name="EMAGenerator")
        self.update_mvag_generator(initial=True)

        if config["AUGMENT"]:
            self.Aug = DiffAug({"colour": True, "translation": True, "cutout": True})
        else:
            self.Aug = None

        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.n_critic = n_critic
        self.EMA_beta = config["EMA_BETA"]
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

    # @tf.function
    def train_step(self, real_images, scale):
        # Determine labels and size of mb for each critic training run
        # (size of real_images = minibatch size * number of critic runs)
        mb_size = real_images.shape[0] // self.n_critic

        if self.fade_iter:
            self.Discriminator.alpha = self.fade_count / self.fade_iter
            self.Generator.alpha = self.fade_count / self.fade_iter

        else:
            self.Discriminator.alpha = None
            self.Generator.alpha = None

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
                d_loss += 0.001 * tf.square(d_predictions[mb_size:]) # Drift term
            
                # Gradient penalty if indicated
                if self.GAN_type == "wasserstein-GP" or "progressive":
                    grad_penalty = gradient_penalty(d_real_batch, d_fake_images, self.Discriminator, scale)
                    d_loss += 10 * grad_penalty
            
            d_grads = d_tape.gradient(d_loss, self.Discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(d_grads, self.Discriminator.trainable_variables))

            # Update metrics
            self.metric_dict["d_metric"].update_state(d_loss)

        # Generator training
        noise = tf.random.normal((mb_size, self.latent_dims), dtype=tf.float32)

        # Get gradients from critic predictions of generated fake images and update weights
        with tf.GradientTape() as g_tape:
            g_fake_images = self.Generator(noise, scale, training=True)
            if self.Aug: g_fake_images = self.Aug.augment(g_fake_images)
            g_predictions = self.Discriminator(g_fake_images, scale, training=True)
            g_loss = self.loss_G(g_predictions, tf.zeros_like(g_predictions))
        
        g_grads = g_tape.gradient(g_loss, self.Generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(g_grads, self.Generator.trainable_variables))
        self.update_mvag_generator()

        # Update metric and increment fade count
        self.metric_dict["g_metric"].update_state(g_loss)
        self.fade_count += 1
