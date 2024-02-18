#!/usr/bin/env python3
"""
a function def autoencoder(input_dims, hidden_layers, 
latent_dims, lambtha): that creates a sparse autoencoder
"""
import tensorflow.keras as keras
K = keras

def sparse_autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder.

    Parameters:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.
        lambtha (float): Regularization parameter for L1 regularization on the encoded output.

    Returns:
        tuple: (encoder, decoder, auto)
            encoder (keras.Model): Encoder model.
            decoder (keras.Model): Decoder model.
            auto (keras.Model): Sparse autoencoder model.
    """
    # Define Encoder model
    encoder_inputs = K.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)
    encoded = K.layers.Dense(latent_dims, activation='relu',
                             activity_regularizer=K.regularizers.l1(lambtha))(x)
    encoder = K.models.Model(inputs=encoder_inputs, outputs=encoded)

    # Define Decoder model
    decoder_inputs = K.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = K.layers.Dense(units, activation='relu')(x)
    decoded = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.models.Model(inputs=decoder_inputs, outputs=decoded)

    # Define Autoencoder 
    auto_inputs = K.Input(shape=(input_dims,))
    auto_en_output = encoder(auto_inputs)
    auto_de_output = decoder(auto_en_output)
    auto = K.models.Model(inputs=auto_inputs, outputs=auto_output)

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto

