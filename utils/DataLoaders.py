import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


class ImgLoader:
    def __init__(self, config):
        self.file_path = config["DATA_PATH"]
        dataset_size = config["DATASET_SIZE"]
        self.img_list = os.listdir(self.file_path)
        np.random.shuffle(self.img_list)
        if dataset_size: self.img_list = self.img_list[0:dataset_size]
    
    def data_generator(self, res, aug):
        np.random.shuffle(self.img_list)
        N = len(self.img_list)
        i = 0

        while i < N:
            img = tf.io.read_file(f"{self.file_path}{self.img_list[i]}")
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (res, res))
            img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
            img = (img * 2) - 1
            if np.random.rand() > 0.5: img = img[:, :, ::-1, :]
            i += 1

            yield img


class DiffAug:

    """ https://arxiv.org/abs/2006.10738
        https://github.com/mit-han-lab/data-efficient-gans """
    
    def __init__(self, aug_config):
        self.aug_config = aug_config

    # Random brightness in range [-0.5, 0.5]
    def brightness(self, x):
        factor = tf.random.uniform((x.shape[0], 1, 1, 1)) - 0.5
        return x + factor
    
    # Random saturation in range [0, 2]
    def saturation(self, x):
        factor = tf.random.uniform((x.shape[0], 1, 1, 1)) * 2
        x_mean = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        return (x - x_mean) * factor + x_mean

    # Random contrast in range [0.5, 1.5]
    def contrast(self, x):
        factor = tf.random.uniform((x.shape[0], 1, 1, 1)) + 0.5
        x_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        return (x - x_mean) * factor + x_mean
    
    # Random translation by ratio 0.125
    def translation(self, x, ratio=0.125):
        batch_size = tf.shape(x)[0]
        image_size = tf.shape(x)[1:3]
        shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
        translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
        translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
        grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0, image_size[0] + 1)
        grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0, image_size[1] + 1)
        x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
        x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
        
        return x

    # Random cutout by ratio 0.5
    def cutout(self, x, ratio=0.5):
        batch_size = tf.shape(x)[0]
        image_size = tf.shape(x)[1:3]
        cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
        offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32)
        offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32)
        grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
        cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
        mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
        cutout_grid = tf.maximum(cutout_grid, 0)
        cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
        mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
        x = x * tf.expand_dims(mask, axis=3)
        
        return x
    
    def augment(self, x):
        if self.aug_config["colour"]: x = self.contrast(self.saturation(self.brightness(x)))
        if self.aug_config["translation"]: x = self.translation(x)
        if self.aug_config["cutout"]: x = self.cutout(x)

        return x


if __name__ == "__main__":

    FILE_PATH = "C:/Users/roybo/OneDrive/Documents/CelebFacesSmall/Imgs/Imgs/"
    imgs_list = os.listdir(FILE_PATH)
    MB_SIZE = 4

    TestLoader = ImgLoader({"DATA_PATH": FILE_PATH, "DATASET_SIZE": 4})
    TestAug = DiffAug({"colour": True, "translation": True, "cutout": True})

    train_ds = tf.data.Dataset.from_generator(
        TestLoader.data_generator, args=[64, 1], output_types=tf.float32)

    for img in train_ds.batch(MB_SIZE):
        img = TestAug.augment(img)
        plt.subplot(2, 2, 1)
        plt.imshow(img[0, :, :, :])
        plt.subplot(2, 2, 2)
        plt.imshow(img[1, :, :, :])
        plt.subplot(2, 2, 3)
        plt.imshow(img[2, :, :, :])
        plt.subplot(2, 2, 4)
        plt.imshow(img[3, :, :, :])
        plt.show()
