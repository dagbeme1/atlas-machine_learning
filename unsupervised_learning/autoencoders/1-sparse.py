#!/usr/bin/env python3
"""
contains the sparse functions
"""
import tensorflow.keras as keras  # Import TensorFlow and its Keras module
K = keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Create a sparse autoencoder instance.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in
        the encoder.
        latent_dims (int): Dimensions of the latent space representation.
        lambtha (float): L1 regularization parameter for sparsity.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model.
    """
    # Define the encoder model
    encoder_inputs = K.Input(shape=(input_dims,))  # Create an input layer
    encoder_outputs = encoder_inputs  # Initialize encoder_outputs

    for i in range(len(hidden_layers)):  # Loop through the hidden layers
        encoder_layer = \
            K.layers.Dense(units=hidden_layers[i], activation='relu')
        # Create a dense hidden layer
        # Connect to the previous layer
        encoder_outputs = encoder_layer(encoder_outputs)

    encoder_layer = K.layers.Dense(
        units=latent_dims,
        activation='relu',
        activity_regularizers=K.regularizers.l1(lambtha))
    # Create the final layer for the encoder
    # Connect to the previous layer
    encoder_outputs = encoder_layer(encoder_outputs)

    encoder = K.models.Model(inputs=encoder_inputs, outputs=encoder_outputs)
    # Create the encoder model

    # Define the decoder model
    # Create an input layer for the decoder
    decoder_inputs = K.Input(shape=(latent_dims,))
    decoder_outputs = decoder_inputs  # Initialize decoder_outputs

    # Loop through the hidden layers in reverse
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder_layer = K.layers.Dense(
            units=hidden_layers[i], activation='relu')
        # Create a dense hidden layer
        # Connect to the previous layer
        decoder_outputs = decoder_layer(decoder_outputs)

    decoder_layer = K.layers.Dense(units=input_dims, activation='sigmoid')
    # Create the final layer for the decoder
    # Connect to the previous layer
    decoder_outputs = decoder_layer(decoder_outputs)

    decoder = K.models.Model(inputs=decoder_inputs, outputs=decoder_outputs)
    # Create the decoder model

    # Define the autoencoder
    auto_outputs = encoder(encoder_inputs)  # Pass input through the encoder
    # Pass encoder output through the decoder
    auto_outputs = decoder(auto_outputs)

    auto = K.models.Model(inputs=encoder_inputs, outputs=auto_outputs)
    # Create the full autoencoder model

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss='binary_crossentropy')
    # Compile the model with Adam optimizer and BCE loss

    # Return the encoder, decoder, and autoencoder models
    return encoder, decoder, auto
