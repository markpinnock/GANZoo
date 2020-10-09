import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

sys.path.append('..')

from Networks import discriminatorModel, generatorModel
from utils.DataLoaders import imgPartition, imgLoader
from utils.TrainFuncs import trainStep


FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/GANImages/"
# FILE_PATH = "D:/VAEImages/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/imgs/"
MB_SIZE = 4
EPOCHS = 50
NOISE_DIM = 256
NUM_EX = 16
SEED = tf.random.uniform([NUM_EX, NOISE_DIM], -1, 1)

# train_list, val_list, test_list = imgPartition(FILE_PATH, PARTITION_FILE)
train_list = os.listdir(FILE_PATH)
N = len(train_list)

train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[FILE_PATH, train_list], output_types=tf.float32).batch(MB_SIZE).prefetch(MB_SIZE)

train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[FILE_PATH, train_list], output_types=tf.float32).batch(MB_SIZE).prefetch(MB_SIZE)

Generator = generatorModel()
Discriminator = discriminatorModel()
print(Generator.summary())
print(Discriminator.summary())

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
