import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def genLossDC(fake):
    BCE = keras.losses.BinaryCrossentropy(from_logits=True)
    return BCE(tf.ones_like(fake), fake)


@tf.function
def discLossDC(real, fake):
    BCE = keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = BCE(tf.ones_like(real), real)
    fake_loss = BCE(tf.zeros_like(fake), fake)
    return real_loss + fake_loss


@tf.function
def genLossLS(fake):
    return 0.5 * tf.reduce_mean(tf.pow(fake - 1, 2))


@tf.function
def discLossLS(real, fake):
    return 0.5 * tf.reduce_mean(tf.pow(real - 1, 2)) + 0.5 * tf.reduce_mean(tf.pow(fake, 2))


@tf.function
def trainStep(
    imgs, Generator, Discriminator, GenOptimiser, DiscOptimiser,\
        mb_size, noise_dim, genMetric, discMetric1, discMetric2, discAcc1, discAcc2):

    noise = tf.random.uniform((mb_size, noise_dim), -1, 1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_img = Generator(noise, training=True)
        real_pred = Discriminator(imgs + tf.random.normal(imgs.shape.as_list(), 0, 1), training=True)
        fake_pred = Discriminator(gen_img + tf.random.normal(gen_img.shape.as_list(), 0, 1), training=True)
        
        gen_losses = genLossLS(fake_pred)
        disc_losses = discLossLS(real_pred, fake_pred)

    gen_gradients = gen_tape.gradient(gen_losses, Generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_losses, Discriminator.trainable_variables)
    GenOptimiser.apply_gradients(zip(gen_gradients, Generator.trainable_variables))
    DiscOptimiser.apply_gradients(zip(disc_gradients, Discriminator.trainable_variables))

    genMetric.update_state(tf.ones_like(fake_pred), fake_pred)
    discMetric1.update_state(tf.ones_like(real_pred), real_pred)
    discMetric2.update_state(tf.zeros_like(fake_pred), fake_pred)
    discAcc1.update_state(tf.ones_like(real_pred), real_pred)
    discAcc2.update_state(tf.zeros_like(fake_pred), fake_pred)

    return gen_gradients, disc_gradients
