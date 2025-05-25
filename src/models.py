import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    GRU,
    Conv1D,
    MaxPooling1D,
    concatenate,
    Flatten,
    Reshape,
    Dropout,
    BatchNormalization,
    LeakyReLU,
    Conv2D,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Reshape, UpSampling1D


def inception_module(input_tensor, filters):

    # 1x1 convolution
    conv_1x1 = Conv1D(filters, 1, padding="same", activation="relu")(input_tensor)

    # 3x3 convolution
    conv_3x3 = Conv1D(filters, 3, padding="same", activation="relu")(input_tensor)

    # 5x5 convolution
    conv_5x5 = Conv1D(filters, 5, padding="same", activation="relu")(input_tensor)

    # MaxPooling followed by 1x1 convolution
    max_pool = MaxPooling1D(3, strides=1, padding="same")(input_tensor)
    max_pool = Conv1D(filters, 1, padding="same", activation="relu")(max_pool)

    # Concatenate all the layers
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)

    return output


def inceptDecoder(input_shape, output_shape):
    input_layer = Input(shape=input_shape)

    # Inception Module 1
    x = inception_module(input_layer, 64)

    # GRU Layer(s)
    x = GRU(128, return_sequences=True, activation="relu")(x)
    x = GRU(256, return_sequences=True, activation="relu")(x)
    x = GRU(512, return_sequences=False, activation="relu")(x)

    # Reshape the output to add a time dimension of 1 (necessary for Conv1D)
    x = Reshape((1, 512))(x)

    # Inception Module 2
    x = inception_module(x, 128)

    # Flatten the output from Inception modules and GRU layers
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    output_layer = Dense(output_shape, activation="linear")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


class NeuroInceptDecoder:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def inception_module(self, input_tensor, filters):
        conv_1x1 = Conv1D(filters, 1, padding="same", activation="relu")(input_tensor)
        conv_3x3 = Conv1D(filters, 3, padding="same", activation="relu")(input_tensor)
        conv_5x5 = Conv1D(filters, 5, padding="same", activation="relu")(input_tensor)

        max_pool = MaxPooling1D(3, strides=1, padding="same")(input_tensor)
        max_pool = Conv1D(filters, 1, padding="same", activation="relu")(max_pool)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)

        return output

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        # Inception Module 1
        x = self.inception_module(input_layer, 64)

        # GRU Module
        x = GRU(128, return_sequences=True, activation="relu")(x)
        x = GRU(256, return_sequences=True, activation="relu")(x)
        x = GRU(512, return_sequences=False, activation="relu")(x)

        x = Reshape((1, 512))(x)

        # Inception Module 2
        x = self.inception_module(x, 128)

        x = Flatten()(x)

        # Fully Connected Layers
        x = Dense(1024, activation="relu")(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)

        output_layer = Dense(self.output_shape, activation="linear")(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        return model


def FCN(input_shape, output_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)  # Flatten layer before the dense blocks

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(output_shape, activation="linear")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def CNN(input_shape, output_shape):
    input_shape = (input_shape[0], input_shape[1], 1)
    inputs = tf.keras.Input(shape=input_shape)  # Adjust input shape as needed

    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = Dropout(0.3)(x)

    # Flatten and Dense layer
    x = Flatten()(x)
    x = Dense(output_shape)(x)

    model = Model(inputs=inputs, outputs=x)

    return model


############ Diffusion Model ############


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv1D, Concatenate, Dense, Flatten, Lambda


# ——— TimeEmbedding layer ———————————————————————
class TimeEmbedding(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, t):
        # t: (batch,1)
        half = self.dim // 2
        factor = tf.math.log(10000.0) / tf.cast(half - 1, tf.float32)
        scaled = tf.cast(t, tf.float32) * tf.exp(
            tf.range(half, dtype=tf.float32) * -factor
        )
        return tf.concat([tf.sin(scaled), tf.cos(scaled)], axis=-1)  # (batch, dim)


# ——— Dilated‐Conv UNet Diffusion Model —————————————
class DilatedUNetDiffusion:
    def __init__(self, input_shape, timesteps=1000, beta_schedule="cosine"):
        self.input_shape = input_shape  # e.g. (9, channels)
        self.timesteps = timesteps
        self.beta_sched = beta_schedule
        self.beta, self.alpha, self.alpha_bar = self._make_noise_schedule()
        self.model = self._build_model()

    def _make_noise_schedule(self):
        if self.beta_sched == "linear":
            b = np.linspace(1e-4, 0.02, self.timesteps)
        else:  # cosine
            steps = np.linspace(0, np.pi / 2, self.timesteps)
            b = (np.cos(steps) ** 2) * 0.02
        a = 1.0 - b
        ab = np.cumprod(a)
        return b, a, ab

    def _build_model(self):
        eeg = Input(shape=self.input_shape, name="eeg_input")  # (batch, 9, C)
        t_in = Input(shape=(1,), name="time_input")  # (batch, 1)

        # time embedding → (batch, dim) → expand and tile to (batch, 9, dim)
        t_emb = TimeEmbedding(self.input_shape[0])(t_in)
        t_emb = Dense(self.input_shape[0], activation="relu")(t_emb)
        t_emb = Lambda(lambda x: tf.expand_dims(x, 1))(t_emb)  # (batch,1,9)
        t_emb = Lambda(lambda x: tf.tile(x, [1, self.input_shape[0], 1]))(t_emb)

        x = Concatenate(axis=-1)([eeg, t_emb])  # (batch,9, C+9)

        # Encoder: dilated convs
        e1 = Conv1D(64, 3, padding="same", dilation_rate=1, activation="relu")(x)
        e2 = Conv1D(128, 3, padding="same", dilation_rate=2, activation="relu")(e1)
        e3 = Conv1D(256, 3, padding="same", dilation_rate=4, activation="relu")(e2)

        # Bottleneck
        b = Conv1D(256, 3, padding="same", dilation_rate=8, activation="relu")(e3)

        # Decoder: mirrored dilations
        d3 = Conv1D(256, 3, padding="same", dilation_rate=4, activation="relu")(b)
        d2 = Conv1D(128, 3, padding="same", dilation_rate=2, activation="relu")(d3)
        d1 = Conv1D(64, 3, padding="same", dilation_rate=1, activation="relu")(d2)

        # Score output (denoising prediction)
        out = Conv1D(self.input_shape[-1], 1, padding="same", name="score_output")(d1)

        return Model(inputs=[eeg, t_in], outputs=out, name="DilatedUNetDiffusion")

    def add_noise(self, x, t):
        # same as before...
        alpha_bar_t = tf.gather(self.alpha_bar, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, (-1, 1, 1))
        noise = tf.random.normal(tf.shape(x))
        return tf.sqrt(alpha_bar_t) * x + tf.sqrt(1 - alpha_bar_t) * noise, noise

    def train_step(self, x, optimizer):
        batch_size = tf.shape(x)[0]
        t = tf.random.uniform((batch_size,), 0, self.timesteps, dtype=tf.int32)
        noisy, true_n = self.add_noise(x, t)
        with tf.GradientTape() as tape:
            pred_n = self.model(
                [noisy, tf.expand_dims(tf.cast(t, tf.float32), -1)], training=True
            )
            loss = tf.reduce_mean(tf.square(true_n - pred_n))
        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def denoise(self, x_init):
        x = x_init
        for t in reversed(range(self.timesteps)):
            t_vec = tf.fill([x.shape[0], 1], float(t))
            pred_n = self.model([x, t_vec], training=False)
            ab_t = self.alpha_bar[t]
            x = (x - tf.sqrt(1 - ab_t) * pred_n) / tf.sqrt(ab_t)
        return x


# ——— Factory wrapper to match your inceptDecoder style —————
def DiffusionUNet(input_shape, output_shape):
    """
    Returns a Keras model that takes only EEG input
    and internally generates timesteps & score predictions,
    then flattens + dense→ output_shape to fit your pipeline.
    """
    # single EEG input
    eeg_in = Input(shape=input_shape, name="eeg_input")
    # generate a random timestep internally
    t_rand = Lambda(
        lambda x: tf.random.uniform((tf.shape(x)[0], 1), 0, 1000, dtype=tf.int32)
    )(eeg_in)

    diff = DilatedUNetDiffusion(input_shape)
    score = diff.model([eeg_in, t_rand])

    flat = Flatten()(score)
    out = Dense(output_shape, activation="linear", name="diffusion_output")(flat)
    return Model(inputs=eeg_in, outputs=out, name="DiffusionUNet")
