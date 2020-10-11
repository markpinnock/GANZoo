import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

sys.path.append('..')

from GAN import GAN
from utils.DataLoaders import imgPartition, dev_img_loader, img_loader

# Dev dataset
IMG_PATH = "C:/Users/roybo/OneDrive/Documents/CelebFacesSmall/Imgs/Imgs/"
# IMG_PATH = "D:/VAEImages/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/imgs/"
# RES = "LS128x128_Small_100"
EXPT = "test"

if not os.path.exists(f"{SAVE_PATH}{EXPT}/"):
    os.mkdir(f"{SAVE_PATH}{EXPT}/")

# Set hyperparameters and example latent sample
MB_SIZE = 64
EPOCHS = 100
LATENT_DIM = 128
NUM_EX = 16
D_NC = 64
G_NC = 64
G_ETA = 2e-4
D_ETA = 2e-4
G_ETA_WS = 5e-5
D_ETA_WS = 5e-5
N_CRITIC = 5

LATENT_SAMPLE = tf.random.normal([NUM_EX, LATENT_DIM], dtype=tf.float32)

# Create dataset
train_list = os.listdir(IMG_PATH)
# train_list = train_list[0:1000]
N = len(train_list)

# Set up dataset with minibatch size multiplied by number of critic training runs
train_ds = tf.data.Dataset.from_generator(
    dev_img_loader, args=[IMG_PATH, train_list], output_types=tf.float32).batch(MB_SIZE * N_CRITIC).prefetch(MB_SIZE)

# Create optimisers and compile model
# GOptimiser = keras.optimizers.Adam(G_ETA, 0.5, 0.999)
# DOptimiser = keras.optimizers.Adam(D_ETA, 0.5, 0.999)
# GOptimiser = keras.optimizers.RMSprop(G_ETA_WS)
# DOptimiser = keras.optimizers.RMSprop(D_ETA_WS)
GOptimiser = keras.optimizers.Adam(1e-4, 0.0, 0.9)
DOptimiser = keras.optimizers.Adam(1e-4, 0.0, 0.9)
Model = GAN(
    latent_dims=LATENT_DIM,
    g_nc=G_NC, d_nc=D_NC,
    g_optimiser=GOptimiser,
    d_optimiser=DOptimiser,
    GAN_type="wasserstein-GP",
    n_critic=N_CRITIC)

# Training loop
for epoch in range(EPOCHS):
    Model.metric_dict["g_metric"].reset_states()
    Model.metric_dict["d_metric_1"].reset_states()
    Model.metric_dict["d_metric_2"].reset_states()

    for imgs in train_ds:
        losses = Model.train_step(imgs)
        # g_loss += losses["g_loss"]
        # d_loss += losses["d_loss"]

    print(f"Ep {epoch + 1}, G: {Model.metric_dict['g_metric'].result():.4f}, D1: {Model.metric_dict['d_metric_1'].result():.4f}, D2: {Model.metric_dict['d_metric_2'].result():.4f}")

    # Generate example images
    if (epoch + 1) % 1 == 0:
        pred = Model.Generator(LATENT_SAMPLE, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(pred.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(pred[i, :, :, :] / 2 + 0.5)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"{SAVE_PATH}{EXPT}/image_at_epoch_{epoch + 1:04d}.png", dpi=250)
        plt.close()

    # Save checkpoint
    # if (epoch + 1) % 10 == 0:
    #     check_path = f"{SAVE_PATH}models/{RES}/"

    #     if not os.path.exists(check_path):
    #         os.mkdir(check_path)
        
    #     G_check_name = f"{SAVE_PATH}models/{RES}/G_{epoch + 1:04d}.ckpt"
    #     D_check_name = f"{SAVE_PATH}models/{RES}/D_{epoch + 1:04d}.ckpt"
    #     Generator.save_weights(G_check_name)
    #     Discriminator.save_weights(D_check_name)