import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from keras import regularizers
from keras.layers import Dense, Input, Layer
from keras.models import Model
from keras.optimizers import Adam

from ivae.layers import InformedBiasConstraint, InformedConstraint
from ivae.utils import set_all_seeds


class VAELoss(Layer):
    """
    Layer that adds VAE total loss to the model.
    """

    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim

    def call(self, inputs):
        inputs, outputs, z_mean, z_log_sigma = inputs
        reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
        reconstruction_loss *= self.input_dim
        # MODIFIED
        z_log_var = tf.clip_by_value(z_log_sigma, -3, 3)

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return outputs


class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        # Modified
        z_log_var = tf.clip_by_value(z_log_var, -3, 3)

        epsilon = tf.random.normal(
            shape=(batch, dim), mean=0.0, stddev=0.1
        )  # , seed=self.seed_generator
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class InformedVAE:
    def __init__(
        self,
        adjacency_matrices,
        adjacency_names,
        adjacency_activation,
        seed,
        learning_rate=1e-5,
    ):
        self.adjacency_matrices = (
            adjacency_matrices
            if isinstance(adjacency_matrices, list)
            else [adjacency_matrices]
        )
        self.adjacency_names = (
            adjacency_names if isinstance(adjacency_names, list) else [adjacency_names]
        )
        self.adjacency_activation = (
            adjacency_activation
            if isinstance(adjacency_activation, list)
            else [adjacency_activation]
        )
        self.latent_dim = self.adjacency_matrices[-1].shape[1] // 2
        # self.act = act
        self.learning_rate = learning_rate
        set_all_seeds(seed)
        self.input_dim = self.adjacency_matrices[0].shape[0]

    def _build_informed_layer(self, adj, name, act):
        return Dense(
            adj.shape[1],
            activation=act,
            activity_regularizer=regularizers.L2(1e-5),
            kernel_constraint=InformedConstraint(adj),
            bias_constraint=InformedBiasConstraint(adj),
            name=name,
        )

    def _build_vae(self):
        self.layers = []
        for adj, name, act in zip(
            self.adjacency_matrices, self.adjacency_names, self.adjacency_activation
        ):
            self.layers.append(self._build_informed_layer(adj, name, act))

        inputs = Input(shape=(self.input_dim,))

        x = inputs
        for layer in self.layers:
            x = layer(x)

        z_mean = Dense(self.latent_dim)(x)
        z_log_sigma = Dense(self.latent_dim)(x)
        z = Sampling()([z_mean, z_log_sigma])

        encoder = Model(inputs, [z_mean, z_log_sigma, z], name="encoder")

        # Decoder
        latent_inputs = Input(shape=(self.latent_dim,), name="z_sampling")
        x = latent_inputs
        for layer, act in zip(reversed(self.layers), self.adjacency_activation):
            x = Dense(layer.units, activation=act)(x)

        outputs = Dense(self.input_dim, activation="linear")(x)
        decoder = Model(latent_inputs, outputs, name="decoder")

        outputs = decoder(encoder(inputs)[2])
        vae_outputs = VAELoss(self.input_dim)([inputs, outputs, z_mean, z_log_sigma])
        vae = Model(inputs, vae_outputs, name="vae_mlp")
        vae.summary()
        optimizer = Adam(learning_rate=self.learning_rate, clipvalue=3.0)
        vae.compile(optimizer=optimizer, metrics=["mse"])

        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def fit(self, *args, **kwargs):
        return self.vae.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.vae.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.vae.evaluate(*args, **kwargs)
