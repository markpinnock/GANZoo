import argparse
import datetime
import json
import numpy as np
import os
import tensorflow as tf

from gan_zoo.training_loops import training_loop
from gan_zoo.networks.progressivegan.ProgGAN import ProgressiveGAN
from gan_zoo.utils.dataloaders import ImgLoader


# Handle arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-cp", help="Config json path", type=str)
arguments = parser.parse_args()

# Parse config json
with open(arguments.config_path, 'r') as infile:
    CONFIG = json.load(infile)

LATENT_SAMPLE = tf.random.normal([CONFIG["EXPT"]["NUM_EXAMPLES"], CONFIG["HYPERPARAMS"]["LATENT_DIM"]], dtype=tf.float32)

# Create optimisers and compile model
if CONFIG["HYPERPARAMS"]["MODEL"] in ["ProgGAN", "StyleGAN"]:
    Model = ProgressiveGAN(config=CONFIG["HYPERPARAMS"])

if CONFIG["HYPERPARAMS"]["OPT"] == "Adam":
    g_opt = tf.keras.optimizers.Adam(*CONFIG["HYPERPARAMS"]["G_ETA"], name="g_opt")
    d_opt = tf.keras.optimizers.Adam(*CONFIG["HYPERPARAMS"]["D_ETA"], name="d_opt")

elif CONFIG["HYPERPARAMS"]["OPT"] == "RMSprop":
    g_opt = tf.keras.optimizers.RMSprop(*CONFIG["HYPERPARAMS"]["G_ETA"], name="g_opt")
    d_opt = tf.keras.optimizers.RMSprop(*CONFIG["HYPERPARAMS"]["D_ETA"], name="d_opt")

Model.compile(g_optimiser=g_opt, d_optimiser=d_opt, loss=CONFIG["HYPERPARAMS"]["LOSS_FN"])

# Print model summary and save graph if necessary
if CONFIG["EXPT"]["VERBOSE"]:
    Model.set_scale(len(CONFIG["EXPT"]["SCALES"]) - 1, 1)
    Model.summary()

if CONFIG["EXPT"]["GRAPH"]:
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = CONFIG["EXPT"]["SAVE_PATH"] + "/logs/" + curr_time
    writer = tf.summary.create_file_writer(log_dir)

    @tf.function
    def trace(x):
        return Model.Discriminator(Model.Generator(x))

    tf.summary.trace_on(graph=True)
    trace(tf.zeros((1, CONFIG["HYPERPARAMS"]["LATENT_DIM"])))

    with writer.as_default():
        tf.summary.trace_export("graph", step=0)
    exit()

# Set up dataset with minibatch size multiplied by number of critic training runs
if CONFIG["EXPT"]["DATASET_SIZE"] == 0:
    CONFIG["EXPT"]["DATASET_SIZE"] = len(os.listdir(CONFIG["EXPT"]["DATA_PATH"]))

DataLoader = ImgLoader(CONFIG["EXPT"])
res = CONFIG["EXPT"]["SCALES"][0]

if CONFIG["EXPT"]["FROM_RAM"]:
    train_ds = tf.data.Dataset.from_tensor_slices(
        DataLoader.data_loader(res=res)
        ).shuffle(CONFIG["EXPT"]["DATASET_SIZE"]).batch(CONFIG["EXPT"]["MB_SIZE"][0] * CONFIG["HYPERPARAMS"]["N_CRITIC"]).prefetch(1)
else:
    train_ds = tf.data.Dataset.from_generator(
        DataLoader.data_generator,
        args=[res], output_shapes=(res, res, 3), output_types=tf.float32
        ).repeat().batch(CONFIG["EXPT"]["MB_SIZE"][0] * CONFIG["HYPERPARAMS"]["N_CRITIC"]).prefetch(1)

Model = training_loop(CONFIG["EXPT"], idx=0, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=False)

for i in range(1, len(CONFIG["EXPT"]["SCALES"])):
    res = CONFIG["EXPT"]["SCALES"][i]

    if CONFIG["EXPT"]["FROM_RAM"]:
        train_ds = tf.data.Dataset.from_tensor_slices(
            DataLoader.data_loader(res=res)
            ).shuffle(CONFIG["EXPT"]["DATASET_SIZE"]).batch(CONFIG["EXPT"]["MB_SIZE"][i] * CONFIG["HYPERPARAMS"]["N_CRITIC"]).prefetch(1)
    else:
        train_ds = tf.data.Dataset.from_generator(
            DataLoader.data_generator,
            args=[res], output_shapes=(res, res, 3), output_types=tf.float32
            ).repeat().batch(CONFIG["EXPT"]["MB_SIZE"][i] * CONFIG["HYPERPARAMS"]["N_CRITIC"]).prefetch(1)

    Model = training_loop(CONFIG["EXPT"], idx=i, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=True)
    Model = training_loop(CONFIG["EXPT"], idx=i, Model=Model, data=train_ds, latent_sample=LATENT_SAMPLE, fade=False)
