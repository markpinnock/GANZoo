import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

from TrainingLoops import training_loop, trace_graph, print_model_summary
from networks.GANWrapper import ProStyleGAN
from utils.DataLoaders import ImgLoader, DiffAug


""" Based on:
    - Karras et al. Progressive Growing of GANs for Improved Quality, Stability, and Variation
    - https://arxiv.org/abs/1710.10196 """

# TODO: MS-SSIM
# TODO: data aug prob
# TODO: truncation trick
# TODO: blurring
# TODO: style mixing

# Handle arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-cp", help="Config json path", type=str)
arguments = parser.parse_args()

# Parse config json
with open(arguments.config_path, 'r') as infile:
    CONFIG = json.load(infile)

LATENT_SAMPLE = tf.random.normal([CONFIG["EXPT"]["NUM_EXAMPLES"], CONFIG["HYPERPARAMS"]["LATENT_DIM"]], dtype=tf.float32)

# Create optimisers and compile model
if CONFIG["HYPERPARAMS"]["MODEL"] in ["ProGAN", "StyleGAN"]:
    Model = ProStyleGAN(config=CONFIG["HYPERPARAMS"])

# trace_graph(Model.Generator, tf.zeros((1, 128)))
# trace_graph(Model.Discriminator, tf.zeros((1, 64, 64, 3)))
# exit()

if CONFIG["EXPT"]["VERBOSE"]:
    print_model_summary(Model.Generator, CONFIG["HYPERPARAMS"]["MAX_RES"])
    print_model_summary(Model.Discriminator, CONFIG["HYPERPARAMS"]["MAX_RES"])

# Set up dataset with minibatch size multiplied by number of critic training runs
DataLoader = ImgLoader(CONFIG["EXPT"])

if CONFIG["EXPT"]["FROM_RAM"]:
    train_ds = tf.data.Dataset.from_tensor_slices(
        DataLoader.data_loader(res=CONFIG["EXPT"]["SCALES"][0])
        ).batch(CONFIG["EXPT"]["MB_SIZE"][0] * CONFIG["HYPERPARAMS"]["N_CRITIC"])
else:
    train_ds = tf.data.Dataset.from_generator(
        DataLoader.data_generator,
        args=[CONFIG["EXPT"]["SCALES"][0]], output_types=tf.float32
        ).batch(CONFIG["EXPT"]["MB_SIZE"][0] * CONFIG["HYPERPARAMS"]["N_CRITIC"]).prefetch(CONFIG["EXPT"]["MB_SIZE"][0])

Model = training_loop(CONFIG["EXPT"], idx=0, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=False)

for i in range(1, len(CONFIG["EXPT"]["SCALES"])):

    if CONFIG["EXPT"]["FROM_RAM"]:
        train_ds = tf.data.Dataset.from_tensor_slices(
            DataLoader.data_loader(res=CONFIG["EXPT"]["SCALES"][i])
            ).batch(CONFIG["EXPT"]["MB_SIZE"][i] * CONFIG["HYPERPARAMS"]["N_CRITIC"])
    else:
        train_ds = tf.data.Dataset.from_generator(
            DataLoader.data_generator,
            args=[CONFIG["EXPT"]["SCALES"][i]], output_types=tf.float32
            ).batch(CONFIG["EXPT"]["MB_SIZE"][i] * CONFIG["HYPERPARAMS"]["N_CRITIC"]).prefetch(CONFIG["EXPT"]["MB_SIZE"][i])

    Model = training_loop(CONFIG["EXPT"], idx=i, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=True)
    Model = training_loop(CONFIG["EXPT"], idx=i, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=False)
