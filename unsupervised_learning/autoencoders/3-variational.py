#!/usr/bin/env python3
"""
Module contains function for creating variational autoencoder.
"""

import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates variational autoencoder network.

    Args:
        input_dims: Integer containing the dimensions of the model input.
        hidden_layers: List containing the number of nodes for each hidden
        layer in the encoder, respectively.
        latent_dims: Integer containing the dimensions of the latent space
        representation.

    Return: encoder, decoder, autoencoder
        encoder: Encoder model, which should output the latent representation,
        the mean, and the log variance, respectively.
        decoder: Decoder model.
        autoencoder: Full autoencoder model.
    """
    K = keras.backend

    def sampling(args):
        """Samples similar points from latent space."""
        mean, logvar = args
        eps = K.random_normal(shape=K.shape(mean))
        return mean + K.exp(0.5 * logvar) * eps

    Dense = keras.layers.Dense
    Loss = keras.losses.binary_crossentropy

    # Define the encoder model
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    # Build hidden layers for the encoder
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)

    # Calculate mean and log variance for the latent space
    mean = Dense(latent_dims)(x)
    logvar = Dense(latent_dims)(x)
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([mean, logvar])

    # Create the encoder model
    encoder = keras.models.Model(inputs=encoder_inputs, outputs=[z, mean, logvar])

    # Define the decoder model
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    # Build hidden layers for the decoder in reverse order
    for units in reversed(hidden_layers):
        x = Dense(units, activation='relu')(x)

    # Output layer for the decoder
    outputs = Dense(input_dims, activation='sigmoid')(x)

    # Create the decoder model
    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Define the autoencoder
    outputs = decoder(encoder(encoder_inputs)[0])
    autoencoder = keras.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Reconstruction loss for the autoencoder
    reconstruction_loss = Loss(encoder_inputs, outputs)
    reconstruction_loss *= input_dims

    # KL divergence loss for the latent space
    k1_loss = 1 + logvar - K.square(mean) - K.exp(logvar)
    k1_loss = K.sum(k1_loss, axis=-1)
    k1_loss *= -0.5

    # Combine both losses for the final VAE loss
    vae_loss = K.mean(reconstruction_loss + k1_loss)
    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer='adam')

    return encoder, decoder, autoencoder

