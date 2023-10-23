#!/usr/bin/env python3
"""
contains the autoencoder functions
"""
import tensorflow.keras as keras  # Import TensorFlow and its Keras module
K = keras  # Alias K as keras

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
    encoder_inputs = K.Input(shape=(input_dims,))  # Create an input layer
    for i in range(len(hidden_layers)):  # Loop through the hidden layers
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')  # Create a dense hidden layer
        if i == 0:  # If it's the first hidden layer
            outputs = layer(encoder_inputs)  # Connect it to the input
        else:
            outputs = layer(outputs)  # Otherwise, connect it to the previous layer
    layer = K.layers.Dense(units=latent_dims, activation='relu')  # Create the final layer for the encoder
    outputs = layer(outputs)  # Connect it to the previous layer
    encoder = K.models.Model(inputs=encoder_inputs, outputs=outputs)  # Create the encoder model

    # Define the decoder model
    decoder_inputs = K.Input(shape=(latent_dims,))  # Create an input layer for the decoder
    for i in range(len(hidden_layers) - 1, -1, -1):  # Loop through the hidden layers in reverse
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')  # Create a dense hidden layer
        if i == len(hidden_layers) - 1:  # If it's the first hidden layer in the decoder
            outputs = layer(decoder_inputs)  # Connect it to the input
        else:
            outputs = layer(outputs)  # Otherwise, connect it to the previous layer
    layer = K.layers.Dense(units=input_dims, activation='sigmoid')  # Create the final layer for the decoder
    outputs = layer(outputs)  # Connect it to the previous layer
    decoder = K.models.Model(inputs=decoder_inputs, outputs=outputs)  # Create the decoder model

    # Define the autoencoder
    outputs = encoder(encoder_inputs)  # Pass input through the encoder
    outputs = decoder(outputs)  # Pass encoder output through the decoder
    auto = K.models.Model(inputs=encoder_inputs, outputs=outputs)  # Create the full autoencoder model

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss='binary_crossentropy')  # Compile the model with Adam optimizer and BCE loss

    return encoder, decoder, auto  # Return the encoder, decoder, and autoencoder models

