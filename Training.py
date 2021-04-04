import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

from TrainingLoops import NonProgGrowTrainingLoop, ProgGrowTrainingLoop, trace_graph
from networks.Model import DCGAN, ProGAN, StyleGAN
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

# Create optimisers and compile model
if CONFIG["HYPERPARAMS"]["MODEL"] == "DCGAN":
    Model = DCGAN(config=CONFIG["HYPERPARAMS"])
elif CONFIG["HYPERPARAMS"]["MODEL"] in ["ProGAN", "StyleGAN"]:
    Model = ProStyleGAN(config=CONFIG["HYPERPARAMS"])
else:
    raise ValueError

# trace_graph(Model.Generator, tf.zeros((1, 128)))
# trace_graph(Model.Discriminator, tf.zeros((1, 64, 64, 3)))
# exit()

if CONFIG["EXPT"]["VERBOSE"]: Model.print_summary()

# Set up dataset with minibatch size multiplied by number of critic training runs
DataLoader = ImgLoader(CONFIG["EXPT"])

# If not progressive growing
if CONFIG["HYPERPARAMS"]["MODEL"] not in ["ProGAN", "StyleGAN"]:
    Train = NonProgGrowTrainingLoop(Model, CONFIG)

    if CONFIG["EXPT"]["FROM_RAM"]:
        train_ds = tf.data.Dataset.from_tensor_slices(
            DataLoader.data_loader(res=CONFIG["EXPT"]["SCALES"])
            ).batch(CONFIG["EXPT"]["MB_SIZE"] * CONFIG["HYPERPARAMS"]["N_CRITIC"])

    else:
        train_ds = tf.data.Dataset.from_generator(
            DataLoader.data_generator,
            args=[CONFIG["EXPT"]["SCALES"]], output_types=tf.float32
            ).batch(CONFIG["EXPT"]["MB_SIZE"] * CONFIG["HYPERPARAMS"]["N_CRITIC"]).prefetch(CONFIG["EXPT"]["MB_SIZE"])
    
    Train.training_loop(train_ds)

# If progressive growing
else:
    Train = ProgGrowTrainingLoop(Model, CONFIG)

    for i in range(0, len(CONFIG["EXPT"]["SCALES"])):

        if CONFIG["EXPT"]["FROM_RAM"]:
            train_ds = tf.data.Dataset.from_tensor_slices(
                DataLoader.data_loader(res=CONFIG["EXPT"]["SCALES"][i])
                ).batch(CONFIG["EXPT"]["MB_SIZE"][i] * CONFIG["HYPERPARAMS"]["N_CRITIC"])
        else:
            train_ds = tf.data.Dataset.from_generator(
                DataLoader.data_generator,
                args=[CONFIG["EXPT"]["SCALES"][i]], output_types=tf.float32
                ).batch(CONFIG["EXPT"]["MB_SIZE"][i] * CONFIG["HYPERPARAMS"]["N_CRITIC"]).prefetch(CONFIG["EXPT"]["MB_SIZE"][i])

        if i > 0:
            Train.training_loop(data=train_ds, idx=i, fade=True)

        Train.training_loop(data=train_ds, idx=i, fade=False)
