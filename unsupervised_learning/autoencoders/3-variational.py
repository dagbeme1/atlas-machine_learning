#!/usr/bin/env python3
"""
a function def autoencoder(input_dims, hidden_layers, latent_dims):
that creates a variational autoencoder:

input_dims is an integer containing the dimensions of the model input
hidden_layers is a list containing the number of nodes for
each hidden layer in the encoder, respectively
the hidden layers should be reversed for the decoder
latent_dims is an integer containing the dimensions of
the latent space representation
Returns: encoder, decoder, auto
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a VAE instance.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): The number of nodes for each hidden layer in
            the encoder, respectively.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full VAE model.
    """
    # Define the encoder model
    encoder_inputs = K.Input(shape=(input_dims,))  # Create an input layer
    x = encoder_inputs  # Initialize x with encoder_inputs
    i = 0  # Initialize i for the while loop
    while i < len(hidden_layers):
        # Add a Dense layer with ReLU activation
        x = K.layers.Dense(units=hidden_layers[i], activation='relu')(x)
        i += 1  # Increment i for the next layer

    # Reparameterization trick
    z_mean = K.layers.Dense(units=latent_dims)(x)  # Compute z_mean
    z_log_var = K.layers.Dense(units=latent_dims)(x)  # Compute z_log_var

    def sampling(args):
        """Sample z."""
        z_mean, z_log_var = args
        epsilon = K.backend.random_normal(shape=(K.backend.shape(z_mean)[0],
                                                 latent_dims))
        z = z_mean + K.backend.exp(0.5 * z_log_var) * epsilon
        return z

    z = K.layers.Lambda(sampling,
                        output_shape=(latent_dims,))([z_mean, z_log_var])
    # Create the encoder model
    encoder = K.models.Model(inputs=encoder_inputs,
                             outputs=[z, z_mean, z_log_var])

    # Define the decoder model
    # Create an input layer for the decoder
    decoder_inputs = K.Input(shape=(latent_dims,))
    x = decoder_inputs  # Initialize x with decoder_inputs
    i = len(hidden_layers) - 1  # Initialize i for the while loop
    while i >= 0:
        # Add a Dense layer with ReLU activation
        x = K.layers.Dense(units=hidden_layers[i], activation='relu')(x)
        i -= 1  # Decrement i for the next layer
    # Final layer with sigmoid activation
    x = K.layers.Dense(units=input_dims, activation='sigmoid')(x)
    # Create the decoder model
    decoder = K.models.Model(inputs=decoder_inputs, outputs=x)

    # Define the autoencoder
    # Pass input through the encoder
    z, z_mean, z_log_var = encoder(encoder_inputs)
    x = decoder(z)  # Pass encoder output through the decoder
    # Create the full VAE model
    auto = K.models.Model(inputs=encoder_inputs, outputs=x)

    def vae_loss(x, x_decoded_mean):
        """VAE loss."""
        reconstruction_loss = K.backend.binary_crossentropy(x, x_decoded_mean)
        reconstruction_loss = K.backend.sum(reconstruction_loss, axis=1)
        kl_loss = -0.5 * K.backend.sum(1
                                       + z_log_var
                                       - K.backend.square(z_mean)
                                       - K.backend.exp(z_log_var), axis=-1)
        return reconstruction_loss + kl_loss

    # Compile the VAE
    # Compile the model with Adam optimizer and VAE loss
    auto.compile(optimizer='Adam', loss=vae_loss)

    return encoder, decoder, auto  # Return the encoder, decoder,& VAE models
