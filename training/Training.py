import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf


""" Based on:
    - Karras et al. Progressive Growing of GANs for Improved Quality, Stability, and Variation
    - https://arxiv.org/abs/1710.10196 """

# TODO: MS-SSIM
# TODO: Equalised learning rate
# TODO: channel numbers
# TODO: runn avg gen weights

sys.path.append("C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/scripts/")

from training_loops import Pix2Pix_training_loop, trace_graph, print_model_summary
from networks.GANWrapper import GAN
from utils.DataLoaders import imgPartition, dev_img_loader, img_loader

# Dev dataset
# IMG_PATH = "C:/Users/roybo/OneDrive/Documents/CelebFacesSmall/Imgs/Imgs/"
IMG_PATH = "D:/Imgs/"
# IMG_PATH = "D:/VAEImages/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/imgs/"
# RES = "LS128x128_Small_100"
EXPT = "test"

if not os.path.exists(f"{SAVE_PATH}{EXPT}/"):
    os.mkdir(f"{SAVE_PATH}{EXPT}/")

# Set hyperparameters and example latent sample
RESOLUTION = 128
NUM_EX = 16
NDF = 16
NGF = 16
GAN_TYPE = "progressive"
SAVE_CKPT = False
# TODO: Convert to argparse

# Set GAN-specific optimisers/hyperparameters
OPT_DICT = {
    "original": {
        "G_OPT": keras.optimizers.Adam(2e-4, 0.5, 0.999),
        "D_OPT": keras.optimizers.Adam(2e-4, 0.5, 0.999),
        "N_CRITIC": 1
        },
    "least_square": {
        "G_OPT": keras.optimizers.Adam(2e-4, 0.5, 0.999),
        "D_ETA": keras.optimizers.Adam(2e-4, 0.5, 0.999),
        "N_CRITIC": 1
        },
    "wasserstein": {
        "G_OPT": keras.optimizers.RMSprop(5e-5),
        "D_OPT": keras.optimizers.RMSprop(5e-5),
        "N_CRITIC": 5
        },
    "wasserstein-GP": {
        "G_OPT": keras.optimizers.Adam(1e-4, 0.0, 0.9),
        "D_OPT": keras.optimizers.Adam(1e-4, 0.0, 0.9),
        "N_CRITIC": 5
    },
    "progressive": {
        "G_OPT": keras.optimizers.Adam(1e-3, 0.0, 0.9),
        "D_OPT": keras.optimizers.Adam(1e-3, 0.0, 0.9),
        "N_CRITIC": 1
    }
}

LATENT_SAMPLE = tf.random.normal([NUM_EX, RESOLUTION], dtype=tf.float32)

# Create dataset
train_list = os.listdir(IMG_PATH)
# train_list = train_list[0:64]
N = len(train_list)

# Create optimisers and compile model
Model = GAN(
    resolution=RESOLUTION,
    g_nc=NGF, d_nc=NDF,
    g_optimiser=OPT_DICT[GAN_TYPE]["G_OPT"],
    d_optimiser=OPT_DICT[GAN_TYPE]["D_OPT"],
    GAN_type=GAN_TYPE,
    n_critic=OPT_DICT[GAN_TYPE]["N_CRITIC"]
    )

# trace_graph(Model.Generator, tf.zeros((1, 128)))
# trace_graph(Model.Discriminator, tf.zeros((1, 64, 64, 3)))
# print_model_summary(Model.Generator, RESOLUTION)
# print_model_summary(Model.Discriminator, RESOLUTION)
# exit()
MB_SIZES = [128, 128, 64, 16, 8]
SCALES = [4, 8, 16, 32, 64]
EPOCHS = [5, 8, 8, 10, 10]

# Set up dataset with minibatch size multiplied by number of critic training runs
train_ds = tf.data.Dataset.from_generator(
    dev_img_loader, args=[IMG_PATH, train_list], output_types=tf.float32).batch(MB_SIZES[0] * OPT_DICT[GAN_TYPE]["N_CRITIC"]).prefetch(MB_SIZES[0])

Model = Pix2Pix_training_loop(MB_SIZES[0], EPOCHS[0], Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, scale=SCALES[0], fade=False)

for i in range(1, len(MB_SIZES)):
    train_ds = tf.data.Dataset.from_generator(
        dev_img_loader, args=[IMG_PATH, train_list], output_types=tf.float32).batch(MB_SIZES[i] * OPT_DICT[GAN_TYPE]["N_CRITIC"]).prefetch(MB_SIZES[i])

    Model = Pix2Pix_training_loop(MB_SIZES[i], EPOCHS[i], Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, scale=SCALES[i], fade=True)
    Model = Pix2Pix_training_loop(MB_SIZES[i], EPOCHS[i], Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, scale=SCALES[i], fade=False)
