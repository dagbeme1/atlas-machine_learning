#!/usr/bin/env python3
"""
Sparse Autoencoder Function
"""
import tensorflow.keras as keras
K = keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder.

    Parameters:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden
        layer in the encoder, respectively.
        latent_dims (int): Dimensions of the latent space representation.
        lambtha (float): Regularization parameter for L1 regularization.

    Returns:
        tuple: (encoder, decoder, auto)
            encoder (keras.Model): Encoder model.
            decoder (keras.Model): Decoder model.
            auto (keras.Model): Sparse autoencoder model.
    """
    # Define ENCODER model with L1 regularization
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    encoded = keras.layers.Dense(latent_dims, activation='relu',
                                 activity_regularizer=K.regularizers.l1(lambtha))(x)
    encoder = keras.Model(inputs=inputs, outputs=encoded)

    # Define DECODER model
    inputs_dec = keras.Input(shape=(latent_dims,))
    x = inputs_dec
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=inputs_dec, outputs=decoded)

    # Define AUTOENCODER model
    auto_bottleneck = encoder.layers[-1].output
    auto_output = decoder(auto_bottleneck)
    auto = keras.Model(inputs=inputs, outputs=auto_output)

    # Compilation of the sparse autoencoder
    auto.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

    return encoder, decoder, auto

