import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


def training_loop(config, idx, Model, data, latent_sample, fade=False):
    SCALE = config["SCALES"][idx]
    EPOCHS = config["EPOCHS"][idx]

    IMG_SAVE_PATH = f"{config['SAVE_PATH']}images/{config['EXPT_NAME']}/"
    if not os.path.exists(IMG_SAVE_PATH): os.mkdir(IMG_SAVE_PATH)
    LOG_SAVE_PATH = f"{config['SAVE_PATH']}logs/{config['EXPT_NAME']}/"
    if not os.path.exists(LOG_SAVE_PATH): os.mkdir(LOG_SAVE_PATH)
    MODEL_SAVE_PATH = f"{config['SAVE_PATH']}models/{config['EXPT_NAME']}/"
    if not os.path.exists(IMG_SAVE_PATH): os.mkdir(MODEL_SAVE_PATH)

    if fade:
        num_batches = config["DATASET_SIZE"] // config["MB_SIZE"][idx]
        num_iter = num_batches * EPOCHS
    else:
        num_iter = 0
    
    Model.fade_set(num_iter)
    scale_idx = int(np.log2(SCALE / config["SCALES"][0]))
    Model.set_trainable_layers(scale_idx)

    for epoch in range(EPOCHS):

        Model.g_metric.reset_states()
        Model.d_metric.reset_states()

        for imgs in data:
            if np.random.rand() > 0.5: imgs = imgs[:, :, ::-1, :]
            _ = Model.train_step(imgs, scale=scale_idx)

        print(f"Scale {SCALE} Fade {fade} Ep {epoch + 1}, G: {Model.g_metric.result():.4f}, D: {Model.d_metric.result():.4f}")

        # Generate example images
        if (epoch + 1) % 1 == 0 and not fade:
            pred = Model.EMAGenerator(latent_sample, scale=scale_idx, training=False)
            pred = Model.Generator(latent_sample, scale=scale_idx, training=False)
            if config["G_OUT"] == "linear": pred = np.clip(pred, -1, 1)

            fig = plt.figure(figsize=(4, 4))

            for i in range(pred.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow(pred[i, :, :, :] / 2 + 0.5)
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"{IMG_SAVE_PATH}{scale_idx}_scale_{SCALE}_epoch_{epoch + 1:02d}.png", dpi=250)
            plt.close()

        # Save checkpoint
        # if (epoch + 1) % 10 == 0 and SAVE_CKPT:
        #     check_path = f"{SAVE_PATH}models/{RES}/"

        #     if not os.path.exists(check_path):
        #         os.mkdir(check_path)
            
        #     G_check_name = f"{SAVE_PATH}models/{RES}/G_{epoch + 1:04d}.ckpt"
        #     D_check_name = f"{SAVE_PATH}models/{RES}/D_{epoch + 1:04d}.ckpt"
        #     Generator.save_weights(G_check_name)
        #     Discriminator.save_weights(D_check_name)
    
    return Model