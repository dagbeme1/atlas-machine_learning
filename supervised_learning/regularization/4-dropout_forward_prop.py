#!/usr/bin/env python3

"""
Forward Propagation with Dropout
"""

import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that conducts forward propagation using Dropout.

    Arguments:
    - X: Input data, a numpy array.
    - weights: Dictionary containing the weights and biases of the neural network.
    - L: Integer representing the number of layers in the neural network.
    - keep_prob: Probability of keeping a neuron active during dropout.

    Returns:
    - cache: Dictionary containing the intermediate values computed during forward propagation.
    """

    # Initialize the cache dictionary to store intermediate values
    cache = {}

    # Store the input data in the cache with key 'A0'
    cache['A0'] = X

    # Perform forward propagation for each layer
    for i in range(L):
        # Compute the linear transformation of the input
        Zi = np.matmul(weights['W' + str(i + 1)], 
                       cache['A' + str(i)]) + weights['b' + str(i + 1)]

        # Apply the appropriate activation function based on the layer index
        if i == L - 1:
            # For the last layer, apply the softmax activation function
            cache['A' + str(i + 1)] = 
            np.exp(Zi) / np.sum(np.exp(Zi), axis=0, keepdims=True)
        else:
            # For other layers, apply
            # the hyperbolic tangent (tanh) activation function
            cache['A' + str(i + 1)] = np.tanh(Zi)

            # Apply dropout by randomly setting 
            # a fraction of the values in the layer to zero
            boolean = np.random.rand
            (cache['A' + str(i + 1)].shape[0], 
             cache['A' + str(i + 1)].shape[1]) < keep_prob
            drop = np.where(boolean == 1, 1, 0)
            cache['A' + str(i + 1)] *= drop
            cache['A' + str(i + 1)] /= keep_prob

            # Store the dropout mask in 
            # the cache for later use in backpropagation
            cache['D' + str(i + 1)] = drop

    # Return the computed intermediate values stored in the cache
    return cache
