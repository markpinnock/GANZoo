import numpy as np
import os
import tensorflow as tf


def imgLoader(file_path, img_list):
    file_path = file_path.decode("utf-8")
    ACE_path = f"{file_path}ACE/"
    NCE_path = f"{file_path}NCE/"
    temp_list = list(zip(ACE_list, NCE_list))
    np.random.shuffle(temp_list)
    ACE_list, NCE_list = zip(*temp_list)
    N = len(ACE_list)
    i = 0

    while i < N:
        try:
            NCE_name = NCE_list[i].decode("utf-8")
            NCE_vol = np.load(NCE_path + NCE_name).astype(np.float32)

            ACE_name = ACE_list[i].decode("utf-8")
            # ACE_name = NCE_name[:-7] + "ACE.npy"
            ACE_vol = np.load(ACE_path + ACE_name).astype(np.float32)

        except Exception as e:
            print(f"IMAGE LOAD FAILURE: {NCE_name} {ACE_name} ({e})")

        else:
            ACE_vol = (ACE_vol - ACE_vol.min()) / (ACE_vol.max() - ACE_vol.min()) * 2 - 1
            NCE_vol = (NCE_vol - NCE_vol.min()) / (NCE_vol.max() - NCE_vol.min()) * 2 - 1
            ACE_vol = ACE_vol[::2, ::2, :, np.newaxis]
            NCE_vol = NCE_vol[::2, ::2, :, np.newaxis]
            yield ACE_vol, NCE_vol

        finally:
            i += 1


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

    # FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/Train/"
    # imgs_list = os.listdir(FILE_PATH)
    # MB_SIZE = 4

    # train_ds = tf.data.Dataset.from_generator(
    #     imgLoader, args=[FILE_PATH, imgs_list], output_types=tf.float32)
    
    # for img in train_ds.batch(MB_SIZE):
    #     print(img.numpy().min(), img.numpy().max())
    #     # pass

    FILE_PATH = "Z:/Virtual_Contrast_Data/"
    ACE_path = f"{FILE_PATH}/ACE/"
    NCE_path = f"{FILE_PATH}/NCE/"
    ACE_imgs = os.listdir(ACE_path)
    NCE_imgs = os.listdir(NCE_path)
    ACE_imgs.sort()
    NCE_imgs.sort()

    N = len(ACE_imgs)
    NUM_FOLDS = 5
    FOLD = 0
    MB_SIZE = 8
    random.seed(10)

    for i in range(N):
        # print(ACE_imgs[i], NCE_imgs[i])
        assert ACE_imgs[i][:-16] == NCE_imgs[i][:-16] and ACE_imgs[i][-11:-8] == NCE_imgs[i][-11:-8], "HI/LO PAIRS DON'T MATCH"

    temp_list = list(zip(ACE_imgs, NCE_imgs))
    random.shuffle(temp_list)
    ACE_imgs, NCE_imgs = zip(*temp_list)

    for i in range(N):
        # print(ACE_imgs[i], NCE_imgs[i])
        assert ACE_imgs[i][:-16] == NCE_imgs[i][:-16] and ACE_imgs[i][-11:-8] == NCE_imgs[i][-11:-8], "HI/LO PAIRS DON'T MATCH"

    train_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[FILE_PATH, ACE_imgs, NCE_imgs], output_types=(tf.float32, tf.float32))

    for data in train_ds.batch(MB_SIZE):
        print(data[0].shape, data[1].shape, data[0].numpy().min(), data[0].numpy().max(), data[1].numpy().min(), data[1].numpy().max())
        # pass
