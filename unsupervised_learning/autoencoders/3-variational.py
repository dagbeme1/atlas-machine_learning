#!/usr/bin/env python3

"""
a function def autoencoder(input_dims, hidden_layers, latent_dims): that creates a variational autoencoder
"""

import tensorflow.keras as keras
K = keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    '
