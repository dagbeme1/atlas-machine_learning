#!/usr/bin/env python3
"""
a function def autoencoder(input_dims, hidden_layers,
latent_dims): that creates an autoencoder
"""

import tensorflow.keras as keras
K = keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder.

    Parameters:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden 
        layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
            encoder (keras.Model): Encoder model.
            decoder (keras.Model): Decoder model.
            auto (keras.Model): Full autoencoder model.
    """

    # Define ENCODER model
    inputs = keras.Input(shape=(input_dims,))

    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    encoded = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs, encoded)

    # Define DECODER model
    inputs_dec = keras.Input(shape=(latent_dims,))
    x = inputs_dec

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs_dec, decoded)

    # Define AUTOENCODER
    auto_en_output = encoder.layers[-1].output
    auto_de_output = decoder(auto_en_output)

    auto = keras.Model(inputs, auto_output)

    # Compilation of AUTOENCODER
    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss='binary_crossentropy')

    return encoder, decoder, auto
