#!/usr/bin/env python3
"""
Layers in a neural network
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Create a layer in a neural network.

    Arguments:
    prev -- tensor output of the previous layer
    n -- number of nodes in the layer to create
    activation -- activation function to be used

    Returns:
    Tensor output of the layer
    """

    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name='layer')
    return layer(prev)
