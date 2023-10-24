#!/usr/bin/env python3
"""
Contains the autoencoder functions.
"""
# Import TensorFlow and its Keras module
import tensorflow.keras as keras
K = keras  # Alias K as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder instance.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each
        hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model.
    """
    # Define the encoder model
    encoder_inputs = K.Input(shape=(input_dims,))  # Create an input layer
    for i in range(len(hidden_layers)):  # Loop through the hidden layers
        # Create a dense hidden layer
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == 0:  # If it's the first hidden layer
            outputs = layer(encoder_inputs)  # Connect it to the input
        else:
            outputs = layer(outputs)  # Connect it to the previous layer
    # Create the final layer for the encoder
    layer = K.layers.Dense(units=latent_dims, activation='relu')
    outputs = layer(outputs)  # Connect it to the previous layer
    # Create the encoder model
    encoder = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Define the decoder model
    # Create an input layer for the decoder
    decoder_inputs = K.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        # Create a dense hidden layer
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
    # If it's the first hidden layer in the decoder
        if i == len(hidden_layers) - 1:
            outputs = layer(decoder_inputs)  # Connect it to the input
        else:
            outputs = layer(outputs)  # Connect it to the previous layer
    # Create the final layer for the decoder
    layer = K.layers.Dense(units=input_dims, activation='sigmoid')
    outputs = layer(outputs)  # Connect it to the previous layer
    # Create the decoder model
    decoder = K.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Define the autoencoder
    # Pass input through the encoder
    outputs = encoder(encoder_inputs)
    # Pass encoder output through the decoder
    outputs = decoder(outputs)
    # Create the full autoencoder model
    auto = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Compile the autoencoder
    # Compile the model with Adam optimizer and BCE loss
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    # Return the encoder, decoder, and autoencoder models
    return encoder, decoder, auto
