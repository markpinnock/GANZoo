import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

sys.path.append('..')

from GAN import GAN
from utils.DataLoaders import imgPartition, dev_img_loader, img_loader
from utils.TrainFuncs import trainStep

# Dev dataset
IMG_PATH = "C:/Users/roybo/OneDrive/Documents/CelebFacesSmall/Imgs/Imgs/"
# IMG_PATH = "D:/VAEImages/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/imgs/"

# Set hyperparameters and example latent sample
MB_SIZE = 64
EPOCHS = 50
LATENT_DIM = 128
NUM_EX = 16
LATENT_SAMPLE = tf.random.normal([NUM_EX, NOISE_DIM], dtype=tf.float32)

train_list = os.listdir(IMG_PATH)
N = len(train_list)

train_ds = tf.data.Dataset.from_generator(
    dev_img_loader, args=[IMG_PATH, train_list], output_types=tf.float32).batch(MB_SIZE).prefetch(MB_SIZE)

Model = Gan(LATENT_DIM)
genTrainMetric = keras.metrics.BinaryCrossentropy(from_logits=True)
discTrainMetric1 = keras.metrics.BinaryCrossentropy(from_logits=True)
discTrainMetric2 = keras.metrics.BinaryCrossentropy(from_logits=True)

GenOptimiser = keras.optimizers.Adam(2e-4, 0.5, 0.999)
DiscOptimiser = keras.optimizers.Adam(2e-4, 0.5, 0.999)

for epoch in range(EPOCHS):
    for imgs in train_ds:
        gen_grad, disc_grad = trainStep(
            imgs, Generator, Discriminator, GenOptimiser, DiscOptimiser,
            MB_SIZE, NOISE_DIM, genTrainMetric, discTrainMetric1,
            discTrainMetric2, discTrainAcc1, discTrainAcc2)

    gen_grad_norm = tf.linalg.global_norm(gen_grad)
    disc_grad_norm = tf.linalg.global_norm(disc_grad)

    # print("==============================================================================")
    print(f"Ep {epoch + 1}, Gen Loss {genTrainMetric.result():.4f} Disc Loss 1 {discTrainMetric1.result():.4f} "\
        f"Disc Loss 2 {discTrainMetric2.result():.4f} Disc Acc 1 {discTrainAcc1.result():.4f} Disc Acc 2 {discTrainAcc2.result():.4f} "\
        f"Gen Grad {gen_grad_norm} Disc Grad {disc_grad_norm}")
    # print("==============================================================================")

    genTrainMetric.reset_states()
    gen_MAE = 0
    discTrainMetric1.reset_states()
    discTrainMetric2.reset_states()

    pred = Generator(SEED, training=True)

    fig = plt.figure(figsize=(4,4))

    for i in range(pred.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(pred[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # plt.savefig(SAVE_PATH + 'image_at_epoch_{:04d}.png'.format(epoch + 1), dpi=250)
    # plt.close()
    plt.show()
