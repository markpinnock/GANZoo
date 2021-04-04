import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from abc import ABC, abstractmethod


def trace_graph(model, input_zeros):

    @tf.function
    def trace(x):
        return model(x, 4)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/logs/" + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    tf.summary.trace_on(graph=True)
    trace(input_zeros)

    with summary_writer.as_default():
        tf.summary.trace_export("graph", step=0)


class BaseTrainingLoop(ABC):

    def __init__(self, Model: object, config: dict, latent_sample=None):
        self.Model = Model
        self.CONFIG = config
        self.SAVE_EVERY = config["EXPT"]["SAVE_EVERY"]

        self.IMG_SAVE_PATH = f"{self.CONFIG['EXPT']['SAVE_PATH']}images/{self.CONFIG['EXPT']['EXPT_NAME']}/"
        if not os.path.exists(self.IMG_SAVE_PATH): os.makedirs(self.IMG_SAVE_PATH)
        self.LOG_SAVE_PATH = f"{self.CONFIG['EXPT']['SAVE_PATH']}logs/{self.CONFIG['EXPT']['EXPT_NAME']}/"
        if not os.path.exists(self.LOG_SAVE_PATH): os.makedirs(self.LOG_SAVE_PATH)
        self.MODEL_SAVE_PATH = f"{self.CONFIG['EXPT']['SAVE_PATH']}models/{self.CONFIG['EXPT']['EXPT_NAME']}/"
        if not os.path.exists(self.MODEL_SAVE_PATH): os.makedirs(self.MODEL_SAVE_PATH)

        if latent_sample:
            self.LATENT_SAMPLE = latent_sample
        else:
            self.LATENT_SAMPLE = tf.random.normal([self.CONFIG["EXPT"]["NUM_EXAMPLES"], self.CONFIG["HYPERPARAMS"]["LATENT_DIM"]], dtype=tf.float32)
        
    @abstractmethod
    def training_loop(self):
        raise NotImplementedError

    @abstractmethod
    def save_images(self, epoch=None, tuning_path=None):
        try:
            pred = self.Model.EMAGenerator(self.LATENT_SAMPLE, training=False)
        except AttributeError:
            pred = self.Model.Generator(self.LATENT_SAMPLE, training=False)

        if self.CONFIG["HYPERPARAMS"]["G_OUT"] == "linear": pred = np.clip(pred, -1, 1)

        fig = plt.figure(figsize=(4, 4))

        for i in range(pred.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(pred[i, :, :, :] / 2 + 0.5)
            plt.axis('off')

        plt.tight_layout()
    
    @abstractmethod
    def save_models(self):
        raise NotImplementedError

    @abstractmethod
    def save_results(self):
        raise NotImplementedError


class NonProgGrowTrainingLoop(BaseTrainingLoop):

    def __init__(self, Model, config):
        super().__init__(Model, config)
        self.MAX_RESOLUTION = self.CONFIG["EXPT"]["SCALES"]
        self.EPOCHS = self.CONFIG["EXPT"]["EPOCHS"]
    
    def training_loop(self, data):
        for epoch in range(self.EPOCHS):
            self.Model.metric_dict["g_metric"].reset_states()
            self.Model.metric_dict["d_metric"].reset_states()

            for imgs in data:
                if np.random.rand() > 0.5: imgs = imgs[:, :, ::-1, :]
                _ = self.Model.train_step(imgs)

            print(f"Ep {epoch + 1}, G: {self.Model.metric_dict['g_metric'].result():.4f}, D: {self.Model.metric_dict['d_metric'].result():.4f}")

            if (epoch + 1) % 1 == 0:
                self.save_images(epoch)

            if self.SAVE_EVERY and (epoch + 1) % self.SAVE_EVERY == 0:
                self.save_models(epoch)
    
    def save_images(self, epoch):
        super().save_images()
        plt.savefig(f"{self.IMG_SAVE_PATH}Epoch_{epoch + 1:02d}.png", dpi=250)
        plt.close()
    
    def save_models(self, epoch):
        G_checkpoint_name = f"{self.MODEL_SAVE_PATH}/G_{epoch + 1:04d}"
        D_checkpoint_name = f"{self.MODEL_SAVE_PATH}/D_{epoch + 1:04d}"
        self.Model.Generator.save_weights(G_checkpoint_name)
        self.Model.Discriminator.save_weights(D_checkpoint_name)

    def save_results(self):
        # TODO
        raise NotImplementedError


class ProgGrowTrainingLoop(BaseTrainingLoop):

    def __init__(self, Model, config):
        super().__init__(Model, config)
        self.SCALES = self.CONFIG["EXPT"]["SCALES"]
        self.EPOCHS = self.CONFIG["EXPT"]["EPOCHS"]


    def training_loop(data, idx, fade=False):
        SCALE = self.SCALES[idx]
        EPOCHS = self.EPOCHS[idx]

        if fade:
            num_batches = self.CONFIG["EXPT"]["DATASET_SIZE"] // self.CONFIG["MB_SIZE"][idx]
            num_iter = num_batches * EPOCHS
        else:
            num_iter = 0
    
        self.Model.fade_set(num_iter)
        scale_idx = int(np.log2(SCALE / self.CONFIG["EXPT"]["SCALES"][0]))
        self.Model.set_trainable_layers(scale_idx)

        for epoch in range(EPOCHS):

            self.Model.metric_dict["g_metric"].reset_states()
            self.Model.metric_dict["d_metric"].reset_states()

            for imgs in data:
                if np.random.rand() > 0.5: imgs = imgs[:, :, ::-1, :]
                _ = self.Model.train_step(imgs, scale=scale_idx)

            print(f"Scale {SCALE} Fade {fade} Ep {epoch + 1}, G: {self.Model.metric_dict['g_metric'].result():.4f}, D: {self.Model.metric_dict['d_metric'].result():.4f}")

            if (epoch + 1) % 1 == 0 and not fade:
                self.save_images(scale_idx, SCALE, epoch)

            if self.SAVE_EVERY and (epoch + 1) % self.SAVE_EVERY == 0:
                self.save_models(scale_idx, SCALE, epoch)
    
    def save_images(self, scale_index, scale, epoch):
        super().save_images()
        savefig(f"{self.IMG_SAVE_PATH}{scale_index}_scale_{scale}_epoch_{epoch + 1:02d}.png", dpi=250)
        close()

    def save_models(self, scale_index, scale, epoch):
        # TODO
        G_checkpoint_name = f"{self.MODEL_SAVE_PATH}/G_{epoch + 1:04d}"
        D_checkpoint_name = f"{self.MODEL_SAVE_PATH}/D_{epoch + 1:04d}"
        self.Model.Generator.save_weights(G_checkpoint_name)
        self.Model.Discriminator.save_weights(D_checkpoint_name)

    def save_results(self):
        # TODO
        raise NotImplementedError
