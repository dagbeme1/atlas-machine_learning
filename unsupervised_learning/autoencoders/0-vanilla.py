#!/usr/bin/env python3
"""
This script contains the definition of an autoencoder using TensorFlow/Keras.
The autoencoder is composed of an encoder and a decoder, both consisting of
dense layers. This autoencoder is used for dimensionality reduction and
representation learning.

The autoencoder can be created by providing the input dimensions, a list of
hidden layer sizes for the encoder, and the number of dimensions in the latent
space representation.
"""
# Import the necessary libraries
import tensorflow.keras as keras
# Alias the imported library
K = keras

# Define the autoencoder function
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
    for i in range(len(hidden_layers)):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == 0:
            outputs = layer(encoder_inputs)
        else:
            outputs = layer(outputs)
    layer = K.layers.Dense(units=latent_dims, activation='relu')
    outputs = layer(outputs)
    encoder = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Define the decoder model
    decoder_inputs = K.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == len(hidden_layers) - 1:
            outputs = layer(decoder_inputs)
        else:
            outputs = layer(outputs)
    layer = K.layers.Dense(units=input_dims, activation='sigmoid')
    outputs = layer(outputs)
    decoder = K.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Define the autoencoder
    outputs = encoder(encoder_inputs)
    outputs = decoder(outputs)
    auto = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
