#!/usr/bin/env python3
"""
a function def autoencoder(input_dims, hidden_layers, latent_dims):
that creates a variational autoencoder:

input_dims is an integer containing the dimensions of the model input
hidden_layers is a list containing the number of nodes
for each hidden layer in the encoder, respectively
the hidden layers should be reversed for the decoder
"""

import tensorflow.keras as keras

def sampling(args):
    """
    Re-parametrization to enable backpropagation.
    Args:
        args: Tuple containing mu (mean from the previous layer) and
        sigma (std from the previous layer).
    Returns:
        z: Sampled distribution.
    """
    # Unpacking
    mu, sigma = args

    # Dimension for the normal distribution (same as z_mean)
    m = keras.backend.shape(mu)[0]
    dims = keras.backend.int_shape(mu)[1]

    # Sampling from a normal distribution with mean=0 and standard deviation=1
    # epsilon ~ N(0,1)
    epsilon = keras.backend.random_normal(shape=(m, dims))

    # Sampled vector
    z = mu + keras.backend.exp(0.5 * sigma) * epsilon

    return z

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder.
    Args:
        input_dims: Integer containing the dimensions of the model input.
        hidden_layers: List containing the number of nodes for each hidden
        layer in the encoder, respectively.
        latent_dims: Integer containing the dimensions of the latent space
        representation.
    Returns:
        encoder: The encoder model, which should output the 
        latent representation, the mean, and the log variance.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))

    my_layer = keras.layers.Dense(units=hidden_layers[0],
                                  activation='relu',
                                  input_shape=(input_dims,))(inputs)

    # Check if there are additional hidden layers
    if len(hidden_layers) > 1:
        for i in range(1, len(hidden_layers)):
            my_layer = keras.layers.Dense(units=hidden_layers[i],
                                          activation='relu')(my_layer)

    mu = keras.layers.Dense(units=latent_dims)(my_layer)
    sigma = keras.layers.Dense(units=latent_dims)(my_layer)

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([mu, sigma])

    encoder = keras.Model(inputs=inputs, outputs=[z, mu, sigma])

    # Decoder
    inputs_dec = keras.Input(shape=(latent_dims,))

    my_layer_dec = keras.layers.Dense(units=hidden_layers[-1],
                                      activation='relu',
                                      input_shape=(latent_dims,))(inputs_dec)

    # Check if there are additional hidden layers
    if len(hidden_layers) > 1:
        for i in range(len(hidden_layers) - 2, -1, -1):
            my_layer_dec = keras.layers.Dense(units=hidden_layers[i],
                                          activation='relu')(my_layer_dec)

    my_layer_dec = keras.layers.Dense(units=input_dims,
                                      activation='sigmoid')(my_layer_dec)

    decoder = keras.Model(inputs=inputs_dec, outputs=my_layer_dec)

    # Autoencoder
    auto_bottleneck = encoder.layers[-1].output
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs, outputs=auto_output)

    def custom_loss(loss_input, loss_output):
        """ Custom loss function """
        # Reconstruction loss
        reconstruction_i = keras.backend.binary_crossentropy(loss_input, loss_output)
        reconstruction_sum = keras.backend.sum(reconstruction_i, axis=1)

        # Kullback–Leibler divergence
        kl_i = keras.backend.square(sigma) + keras.backend.square(mu) -
        keras.backend.log(1e-8 + keras.backend.square(sigma)) - 1

        kl_sum = 0.5 * keras.backend.sum(kl_i, axis=1)

        return reconstruction_sum + kl_sum

    # Compilation
    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss=custom_loss)

    return encoder, decoder, auto
