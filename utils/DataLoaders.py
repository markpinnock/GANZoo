import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


class ImgLoader:
    def __init__(self, config):
        self.file_path = config["DATA_PATH"]
        dataset_size = config["DATASET_SIZE"]
        img_list = os.listdir(self.file_path)
        np.random.shuffle(img_list)
        if dataset_size: self.img_list = img_list[0:dataset_size]
    
    def data_generator(self, res):
        np.random.shuffle(self.img_list)
        N = len(self.img_list)
        i = 0

        while i < N:
            img = tf.io.read_file(f"{self.file_path}{self.img_list[i]}")
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = (img * 2) - 1
            img = tf.image.resize(img, (res, res))
            i += 1
            yield img


if __name__ == "__main__":

    FILE_PATH = "C:/Users/roybo/OneDrive/Documents/CelebFacesSmall/Imgs/Imgs/"
    imgs_list = os.listdir(FILE_PATH)
    MB_SIZE = 4

    TestLoader = ImgLoader({"DATA_PATH": FILE_PATH, "DATASET_SIZE": 8})

    train_ds = tf.data.Dataset.from_generator(
        TestLoader.data_generator, args=[16], output_types=tf.float32)

    for img in train_ds.batch(MB_SIZE):
        plt.subplot(2, 2, 1)
        plt.imshow(img[0, ...])
        plt.subplot(2, 2, 2)
        plt.imshow(img[1, ...])
        plt.subplot(2, 2, 3)
        plt.imshow(img[2, ...])
        plt.subplot(2, 2, 4)
        plt.imshow(img[3, ...])
        plt.show()

    for img in train_ds.batch(MB_SIZE):
        plt.subplot(2, 2, 1)
        plt.imshow(img[0, ...])
        plt.subplot(2, 2, 2)
        plt.imshow(img[1, ...])
        plt.subplot(2, 2, 3)
        plt.imshow(img[2, ...])
        plt.subplot(2, 2, 4)
        plt.imshow(img[3, ...])
        plt.show()