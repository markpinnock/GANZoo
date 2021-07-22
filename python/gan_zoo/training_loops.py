import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras.api._v1.keras import callbacks


class ExampleImages(tf.keras.callbacks.Callback):
    def __init__(self, save_path, idx, scale):
        super().__init__()
        self.save_path = save_path
        self.idx = idx
        self.scale = scale

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model(0) # TODO: use EMA generator
        pred = np.clip(pred, -1, 1)

        fig = plt.figure(figsize=(4, 4))

        for i in range(pred.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(pred[i, :, :, :] / 2 + 0.5)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{self.idx}_scale_{self.scale}_epoch_{epoch + 1:02d}.png", dpi=250)
        plt.close()


def training_loop(config, idx, Model, data, latent_sample, fade=False):
    SCALE = config["SCALES"][idx]
    EPOCHS = config["EPOCHS"][idx]

    IMG_SAVE_PATH = f"{config['SAVE_PATH']}images/{config['EXPT_NAME']}/"
    if not os.path.exists(IMG_SAVE_PATH): os.mkdir(IMG_SAVE_PATH)
    LOG_SAVE_PATH = f"{config['SAVE_PATH']}logs/{config['EXPT_NAME']}/"
    if not os.path.exists(LOG_SAVE_PATH): os.mkdir(LOG_SAVE_PATH)
    MODEL_SAVE_PATH = f"{config['SAVE_PATH']}models/{config['EXPT_NAME']}/"
    if not os.path.exists(IMG_SAVE_PATH): os.mkdir(MODEL_SAVE_PATH)

    num_batches = config["DATASET_SIZE"] // config["MB_SIZE"][idx]

    if fade:
        num_iter = num_batches * EPOCHS
    else:
        num_iter = 0
  
    Model.fade_set(num_iter)
    Model.set_scale(idx, config["MB_SIZE"][idx])

    # ExImageCallback = ExampleImages(IMG_SAVE_PATH, idx, SCALE)
    # Model.fit(x=data, epochs=EPOCHS, steps_per_epoch=num_batches, callbacks=[ExImageCallback])

    # return Model

    for epoch in range(EPOCHS):

        Model.g_metric.reset_states()
        Model.d_metric.reset_states()

        for imgs in data:
            if np.random.rand() > 0.5: imgs = imgs[:, :, ::-1, :]
            _ = Model.train_step(imgs)

        print(f"Scale {SCALE} Fade {fade} Ep {epoch + 1}, G: {Model.g_metric.result():.4f}, D: {Model.d_metric.result():.4f}")

        # Generate example images
        if (epoch + 1) % 1 == 0 and not fade:
            # pred = Model.EMAGenerator(latent_sample, training=False)
            # pred = Model.Generator(latent_sample, training=False)
            pred = Model(0)
            if config["G_OUT"] == "linear": pred = np.clip(pred, -1, 1)

            fig = plt.figure(figsize=(4, 4))

            for i in range(pred.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow(pred[i, :, :, :] / 2 + 0.5)
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"{IMG_SAVE_PATH}{idx}_scale_{SCALE}_epoch_{epoch + 1:02d}.png", dpi=250)
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
