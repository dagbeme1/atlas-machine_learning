#!/usr/bin/env python3
"""
contains the sparse functions
"""

import tensorflow.keras as keras
K = keras

def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Instantiates a sparse autoencoder instance.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): The number of nodes for each hidden layer in
            the encoder, respectively.
        latent_dims (int): The dimensions of the latent space representation.
        lambtha (float): The regularization parameter for sparsity.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model.
    """

    # Define the encoder model
    encoder_inputs = K.Input(shape=(input_dims,))
    encoder_layers = []

    i = 0
    while i < len(hidden_layers):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        encoder_layers.append(layer)
        if i == 0:
            encoder_outputs = layer(encoder_inputs)
        else:
            encoder_outputs = layer(encoder_outputs)
        i += 1

    reg_layer = K.layers.Dense(units=latent_dims, activation='relu',
                              activity_regularizer=K.regularizers.l1(lambtha))
    encoder_layers.append(reg_layer)
    encoder_outputs = reg_layer(encoder_outputs)
    encoder = K.models.Model(inputs=encoder_inputs, outputs=encoder_outputs)

    # Define the decoder model
    decoder_inputs = K.Input(shape=(latent_dims,))
    decoder_layers = []
    i = len(hidden_layers) - 1
    while i >= 0:
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        decoder_layers.append(layer)
        if i == len(hidden_layers) - 1:
            decoder_outputs = layer(decoder_inputs)
        else:
            decoder_outputs = layer(decoder_outputs)
        i -= 1

    output_layer = K.layers.Dense(units=input_dims, activation='sigmoid')
    decoder_layers.append(output_layer)
    decoder_outputs = output_layer(decoder_outputs)
    decoder = K.models.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # Define the autoencoder
    auto_outputs = encoder(encoder_inputs)
    auto_outputs = decoder(auto_outputs)
    auto = K.models.Model(inputs=encoder_inputs, outputs=auto_outputs)

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
