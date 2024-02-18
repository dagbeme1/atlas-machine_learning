#!/usr/bin/env python3
"""
a function def autoencoder(input_dims, hidden_layers, latent_dims)
that creates an autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder model.

    Parameters:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: Tuple containing the encoder, decoder, and full autoencoder models.
    """
    # Encoder
    inputs = Input(shape=(input_dims,))  # Define input layer with specified input dimensions
    x = inputs  # Set initial value for the encoding process
    for units in hidden_layers:  # Iterate through hidden layer sizes
        x = Dense(units, activation='relu')(x)  # Add a dense layer with ReLU activation to the encoder
    encoded = Dense(latent_dims, activation='relu')(x)  # Final encoded layer with ReLU activation

    # Decoder
    x = encoded  # Set initial value for the decoding process
    for units in reversed(hidden_layers):  # Iterate through hidden layer sizes in reverse order
        x = Dense(units, activation='relu')(x)  # Add a dense layer with ReLU activation to the decoder
    decoded = Dense(input_dims, activation='sigmoid')(x)  # Final decoded layer with sigmoid activation

    # Autoencoder
    auto = Model(inputs, decoded)  # Full autoencoder model from input to decoded output
    auto.compile(optimizer='adam', loss='binary_crossentropy')  # Compile the autoencoder model

    return auto, Model(inputs, encoded), Model(encoded, decoded)  # Return the autoencoder, encoder, and decoder models

