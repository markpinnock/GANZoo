import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

sys.path.append('..')

from Pix2PixNetworks import discriminatorModel, generatorModel
from utils.DataLoaders import imgPartition, imgLoader
from utils.TrainFuncs import trainStep


FILE_PATH = "Z:/Virtual_Contrast_Data/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/011_GAN_SISR/imgs/"
MB_SIZE = 1
EPOCHS = 100
D_IMG_DIMS = (256, 256, 12, 1, )
G_IMG_DIMS = (256, 256, 12, 1, )
LAMBDA = 100
D_ETA = 2e-4
G_ETA = 2e-4

ACE_list = os.listdir(f"{FILE_PATH}ACE/")
NCE_list = os.listdir(f"{FILE_PATH}NCE/")
N = len(ACE_list)

ACE_examples = [s.encode("utf-8") for s in ACE_list[0:4]]
NCE_examples = [s.encode("utf-8") for s in NCE_list[0:4]]
ACE_test = np.zeros((4, 256, 256, 12, 1), dtype=np.float32)
NCE_test = np.zeros((4, 256, 256, 12, 1), dtype=np.float32)

for i in range(4):
    for data in imgLoader(FILE_PATH.encode("utf-8"), [ACE_examples[i]], [NCE_examples[i]]):
        ACE_test[i, ...] = data[0]
        NCE_test[i, ...] = data[1]

train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[FILE_PATH, ACE_list, NCE_list], output_types=(tf.float32, tf.float32)).batch(MB_SIZE).prefetch(MB_SIZE)

Generator = generatorModel(G_IMG_DIMS)
Discriminator = discriminatorModel(D_IMG_DIMS)
print(Generator.summary())
print(Discriminator.summary())

genTrainMetric = keras.metrics.BinaryCrossentropy(from_logits=True)
discTrainMetric1 = keras.metrics.BinaryCrossentropy(from_logits=True)
discTrainMetric2 = keras.metrics.BinaryCrossentropy(from_logits=True)

GenOptimiser = keras.optimizers.Adam(G_ETA, 0.5, 0.999)
DiscOptimiser = keras.optimizers.Adam(D_ETA, 0.5, 0.999)

for epoch in range(EPOCHS):
    for ACE_imgs, NCE_imgs in train_ds:
        gen_MAE, gen_grad, disc_grad = trainStep(
            ACE_imgs, NCE_imgs, Generator, Discriminator, GenOptimiser, DiscOptimiser,
            LAMBDA, genTrainMetric, discTrainMetric1, discTrainMetric2)

    gen_grad_norm = tf.linalg.global_norm(gen_grad)
    disc_grad_norm = tf.linalg.global_norm(disc_grad)

    # print("==============================================================================")
    print("Epoch {}, Gen Loss 1 {:.4f} Gen Loss 2 {:.4f} Disc Loss 1 {:.4f}  Disc Loss 2 {:.4f}".format(
        epoch, genTrainMetric.result(), gen_MAE.numpy(), discTrainMetric1.result(), discTrainMetric2.result()))
    # print("==============================================================================")

    genTrainMetric.reset_states()
    gen_MAE = 0
    discTrainMetric1.reset_states()
    discTrainMetric2.reset_states()

    pred = Generator(NCE_test, training=True).numpy()

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))

    for i in range(pred.shape[0]):
        axs[0, i].imshow(np.flipud(NCE_test[i, :, :, 6, 0] - pred[i, :, :, 6, 0]), cmap='gray', origin='lower')
        axs[0, i].axis('off')
        axs[1, i].imshow(np.flipud(pred[i, :, :, 6, 0]), cmap='gray', origin='lower')
        axs[1, i].axis('off')
        axs[2, i].imshow(np.flipud(ACE_test[i, :, :, 6, 0]), cmap='gray', origin='lower')
        axs[2, i].axis('off')
        axs[3, i].imshow(np.flipud(ACE_test[i, :, :, 6, 0] - pred[i, :, :, 6, 0]), cmap='gray', origin='lower')
        axs[3, i].axis('off')

    plt.savefig(SAVE_PATH + 'image_at_epoch_{:04d}.png'.format(epoch), dpi=250)
    plt.close()
