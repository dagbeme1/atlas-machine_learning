#!/usr/bin/env python3
"""
Autoencoder Function
"""
import tensorflow.keras as keras
K = keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
        Creates an alternative autoencoder instance.

    Parameters:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden
        layer in the encoder, respectively.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
            encoder (keras.Model): Encoder model.
            decoder (keras.Model): Decoder model.
            auto (keras.Model): Full autoencoder model.
    """

    # ENCODER
    # Placeholder
    encoder_inputs = K.Input(shape=(input_dims,))
    # Densely-connected layer
    for i in range(len(hidden_layers)):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == 0:
            outputs = layer(encoder_inputs)
        else:
            outputs = layer(outputs)
    layer = K.layers.Dense(units=latent_dims, activation='relu')
    outputs = layer(outputs)
    encoder = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # DECODER
    # Placeholder
    decoder_inputs = K.Input(shape=(latent_dims,))
    # Densely connected layer
    for i in range(len(hidden_layers) - 1, -1, -1):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == len(hidden_layers) - 1:
            outputs = layer(decoder_inputs)
        else:
            outputs = layer(outputs)
    layer = K.layers.Dense(units=input_dims, activation='sigmoid')
    outputs = layer(outputs)
    decoder = K.models.Model(inputs=decoder_inputs, outputs=outputs)

    # AUTOENCODERS
    outputs = encoder(encoder_inputs)
    outputs = decoder(outputs)
    auto = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # COMPILATION
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
