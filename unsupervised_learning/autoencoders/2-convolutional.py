#!/usr/bin/env python3
"""
a function def autoencoder(input_dims, filters, latent_dims):
that creates a convolutional autoencoder
"""

import tensorflow.keras as keras  # Import the necessary library


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): The dimensions of the model input.
        filters (list): The number of filters for each convolutional layer in
            the encoder, respectively.
        latent_dims (tuple): The dimensions of the latent space representation

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model.
    """
    # ENCODER
    input_layer = keras.Input(shape=input_dims)  # Create an input layer
    x = input_layer  # Set x to the input layer
    i = 0  # Initialize i to 0

    # Convolutional layers in the encoder
    while i < len(filters):  # Start a while loop
        x = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                padding='same')(x)  # Add a convolutional layer
        # Add max pooling
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        i += 1  # Increment i

    # Create the encoder model
    encoder = keras.models.Model(inputs=input_layer, outputs=x)
    encoder.summary()  # Print a summary of the encoder

    # DECODER
    # Create an input layer for the latent space
    latent_input = keras.Input(shape=latent_dims)
    x = latent_input  # Set x to the latent input
    i = 0  # Reset i to 0

    # Reverse the filters for the decoder
    filters.reverse()  # Reverse the order of the filters

    # Convolutional layers in the decoder
    while i < len(filters) - 1:  # Start a while loop
        x = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                padding='same')(x)  # Add a convolutional layer
        x = keras.layers.UpSampling2D((2, 2))(x)  # Add upsampling
        i += 1  # Increment i

    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu',
                            # Add the second to last convolution
                            padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)  # Add upsampling
    x = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                            padding='same')(x)  # Add the last convolution

    # Create the decoder model
    decoder = keras.models.Model(inputs=latent_input, outputs=x)
    decoder.summary()  # Print a summary of the decoder

    # AUTOENCODER
    # Create an input layer for the autoencoder
    auto_input = keras.Input(shape=input_dims)
    auto_encoder = encoder(auto_input)  # Pass input through the encoder
    # Pass encoder output through the decoder
    auto_decoder = decoder(auto_encoder)
    # Create the full autoencoder model
    auto = keras.models.Model(inputs=auto_input, outputs=auto_decoder)

    # Compile the autoencoder
    # Compile the model with Adam optimizer and BCE loss
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    # Return the encoder, decoder, and autoencoder models
    return encoder, decoder, auto
