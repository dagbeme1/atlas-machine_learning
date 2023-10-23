#!/usr/bin/env python3
"""
0-vanilla.py
"""
import tensorflow.keras as keras
K = keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder instance.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): The number of nodes for each hidden layer in
            the encoder, respectively.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model.
    """
    # Define the encoder model
    encoder_inputs = K.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)
    latent_space = K.layers.Dense(latent_dims, activation='relu')(x)
    encoder = K.models.Model(inputs=encoder_inputs, outputs=latent_space)

    # Define the decoder model
    decoder_inputs = K.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in hidden_layers[::-1]:
        x = K.layers.Dense(units, activation='relu')(x)
    reconstruction = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.models.Model(inputs=decoder_inputs, outputs=reconstruction)

    # Define the autoencoder
    auto_inputs = K.Input(shape=(input_dims,))
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    auto = K.models.Model(inputs=auto_inputs, outputs=decoded)

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto

