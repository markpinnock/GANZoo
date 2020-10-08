import numpy as np
import os
import tensorflow as tf


def imgLoader(file_path, img_list):
    np.random.shuffle(img_list)
    N = len(img_list)
    i = 0

    while i < N:
        img = np.load(file_path + img_list[i])
        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1
        i += 1
        yield img[::4, ::4, tf.newaxis]


def imgPartition(file_path, partition_file):
    train_list = []
    val_list = []
    test_list = []

    with open(file_path + partition_file, 'r') as f:
        for line in f:
            if line.split(' ')[1] == '0\n':
                train_list.append(line.split(' ')[0])
            elif line.split(' ')[1] == '1\n':
                val_list.append(line.split(' ')[0])
            elif line.split(' ')[1] == '2\n':
                test_list.append(line.split(' ')[0])
            else:
                raise ValueError("Incorrect train/val/test index")

    return train_list, val_list, test_list


if __name__ == "__main__":

    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/Train/"
    imgs_list = os.listdir(FILE_PATH)
    MB_SIZE = 4

    train_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[FILE_PATH, imgs_list], output_types=tf.float32)
    
    for img in train_ds.batch(MB_SIZE):
        print(img.numpy().min(), img.numpy().max())
        # pass
