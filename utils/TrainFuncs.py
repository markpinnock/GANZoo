import tensorflow as tf
import tensorflow.keras as keras


class WeightClipConstraint(keras.constraints.Constraint):
    def __init__(self, clip_val):
        self.clip_val = clip_val
    
    def call(self, weights):
        return keras.backend.clip(weights, -self.clip_val, self.clip_val)
    
    def get_config(self):
        return {"clip_value": self.clip_val}


@tf.function
def least_square_loss(labels, predictions):
    if tf.reduce_sum(labels) > 0:
        fake_loss = 0.5 * tf.reduce_mean(tf.pow(predictions[0:labels.shape[0] // 2] - 1, 2))
        real_loss = 0.5 * tf.reduce_mean(tf.pow(predictions[labels.shape[0] // 2:], 2))
        return fake_loss + real_loss
    else:
        return 0.5 * tf.reduce_mean(tf.pow(predictions, 2))


@tf.function
def wasserstein_loss(labels, predictions):
    return tf.reduce_mean(labels * predictions)