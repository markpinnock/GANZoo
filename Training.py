import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

from TrainingLoops import training_loop, trace_graph, print_model_summary
from networks.GANWrapper import GAN
from utils.DataLoaders import ImgLoader, DiffAug


""" Based on:
    - Karras et al. Progressive Growing of GANs for Improved Quality, Stability, and Variation
    - https://arxiv.org/abs/1710.10196 """

# TODO: MS-SSIM
# TODO: runn avg gen weights

# Handle arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-cp", help="Config json path", type=str)
arguments = parser.parse_args()

# Parse config json
with open(arguments.config_path, 'r') as infile:
    CONFIG = json.load(infile)

# Set GAN-specific optimisers/hyperparameters
# TODO: move into GANWrapper
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

LATENT_SAMPLE = tf.random.normal([CONFIG["EXPT"]["NUM_EXAMPLES"], CONFIG["HYPERPARAMS"]["LATENT_DIM"]], dtype=tf.float32)

# Create optimisers and compile model
Model = GAN(
    config=CONFIG["HYPERPARAMS"],
    g_optimiser=OPT_DICT[CONFIG["HYPERPARAMS"]["MODEL"]]["G_OPT"],
    d_optimiser=OPT_DICT[CONFIG["HYPERPARAMS"]["MODEL"]]["D_OPT"],
    n_critic=OPT_DICT[CONFIG["HYPERPARAMS"]["MODEL"]]["N_CRITIC"]
    )

# trace_graph(Model.Generator, tf.zeros((1, 128)))
# trace_graph(Model.Discriminator, tf.zeros((1, 64, 64, 3)))
if CONFIG["EXPT"]["VERBOSE"]:
    print_model_summary(Model.Generator, CONFIG["HYPERPARAMS"]["MAX_RES"])
    print_model_summary(Model.Discriminator, CONFIG["HYPERPARAMS"]["MAX_RES"])

# Set up dataset with minibatch size multiplied by number of critic training runs
DataLoader = ImgLoader(CONFIG["EXPT"])
train_ds = tf.data.Dataset.from_generator(
    DataLoader.data_generator,
    args=[CONFIG["EXPT"]["SCALES"][0], CONFIG["HYPERPARAMS"]["AUGMENT"]],
    output_types=tf.float32
    ).batch(CONFIG["EXPT"]["MB_SIZE"][0] * OPT_DICT[CONFIG["HYPERPARAMS"]["MODEL"]]["N_CRITIC"]).prefetch(CONFIG["EXPT"]["MB_SIZE"][0])

Model = training_loop(CONFIG["EXPT"], idx=0, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=False)

for i in range(1, len(CONFIG["EXPT"]["SCALES"])):
    train_ds = tf.data.Dataset.from_generator(
        DataLoader.data_generator,
        args=[CONFIG["EXPT"]["SCALES"][i], CONFIG["HYPERPARAMS"]["AUGMENT"]],
        output_types=tf.float32
        ).batch(CONFIG["EXPT"]["MB_SIZE"][0] * OPT_DICT[CONFIG["HYPERPARAMS"]["MODEL"]]["N_CRITIC"]).prefetch(CONFIG["EXPT"]["MB_SIZE"][0])

    Model = training_loop(CONFIG["EXPT"], idx=i, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=True)
    Model = training_loop(CONFIG["EXPT"], idx=i, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=False)
