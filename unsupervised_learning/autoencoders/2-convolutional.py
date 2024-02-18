#!/usr/bin/env python3
"""
a function def autoencoder(input_dims, filters,
latent_dims): that creates a convolutional autoencoder
"""

import tensorflow.keras as keras
K = keras

def autoencoder(input_dims, filters, latent_dims):
    """
    Creates convolutional autoencoder.

    Args:
        input_dims: Tuple of integers - dimensions of the model input.
        filters: List - number of filters for each convolutional layer
        in the encoder, respectively.
        latent_dims: Tuple of integers - dimensions of the latent space
        representation.

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        autoencoder: The full autoencoder model.
    """
    # Define the ENCODER model
    Conv2d = keras.layers.Conv2D
    MaxPool = keras.layers.MaxPool2D
    UpSample = keras.layers.UpSampling2D

    encoder_input = keras.Input(shape=input_dims)
    decoder_input = keras.Input(shape=latent_dims)

    encoded_layer = Conv2d(
        filters[0], (3, 3), activation="relu", padding="same")(encoder_input)
    encoded_layer = MaxPool((2, 2), padding="same")(encoded_layer)

    for filter in filters[1:]:
        encoded_layer = Conv2d(
            filter, (3, 3), padding="same", activation="relu")(encoded_layer)
        encoded_layer = MaxPool((2, 2), padding="same")(encoded_layer)

    encoder = keras.Model(encoder_input, encoded_layer)

    # Define the DECODER model
    last_filter = input_dims[-1]

    my_conv_layer_dec = keras.layers.Conv2D(
        filters=filters[-1], kernel_size=(3, 3),
        padding="same", activation='relu')(decoder_input)

    upsampling_lay = keras.layers.UpSampling2D(size=(2, 2))(my_conv_layer_dec)

    for i in range(len(filters) - 2, -1, -1):
        my_conv_layer_dec = keras.layers.Conv2D(
            filters=filters[i], kernel_size=(3, 3),
            padding="same", activation='relu')(upsampling_lay)

        upsampling_lay = keras.layers.UpSampling2D(size=(2, 2))(my_conv_layer_dec)

    my_conv_layer_dec = keras.layers.Conv2D(
        filters=filters[0], kernel_size=(3, 3),
        padding="valid", activation='relu')(upsampling_lay)

    my_conv_layer_dec = keras.layers.Conv2D(
        filters=last_filter, kernel_size=(3, 3),
        padding="valid", activation='sigmoid')(my_conv_layer_dec)

    decoder = keras.Model(inputs=decoder_input, outputs=my_conv_layer_dec)

    # Define the AUTOENCODER
    auto_en_output = encoder.layers[-1].output
    auto_output = decoder(auto_en_output)

    autoencoder = keras.Model(inputs=encoder_input, outputs=auto_output)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder

