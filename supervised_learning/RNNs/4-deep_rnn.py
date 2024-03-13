#!/usr/bin/env python3
"""
the function def deep_rnn(rnn_cells, X, h_0):
that performs forward propagation for a deep RNN
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Forward propagation for a deep RNN.

    Args:
        rnn_cells (list): List of RNNCell instances representing layers of the deep RNN.
        X (numpy.ndarray): Input data of shape (t, m, i), where:
                           - t is the maximum number of time steps
                           - m is the batch size
                           - i is the dimensionality of the data
        h_0 (numpy.ndarray): Initial hidden state of shape (l, m, h), where:
                              - l is the number of layers
                              - m is the batch size
                              - h is the dimensionality of the hidden state

    Returns:
        tuple: A tuple containing:
            H (numpy.ndarray): All hidden states of shape (t+1, layers, m, h).
                               H[0] represents the initial hidden states.
            Y (numpy.ndarray): All outputs of shape (t, m, o), where o is the dimensionality of the outputs.
    """
    # List to store outputs
    Y = []
    # Extract dimensions
    t, m, i = X.shape
    _, _, h = h_0.shape
    # Range of time steps
    time_step = range(t)
    # Number of layers
    layers = len(rnn_cells)
    # Initialize array to store hidden states
    H = np.zeros((t + 1, layers, m, h))
    # Set initial hidden states
    H[0, :, :, :] = h_0

    # Iterate through each time step
    for ts in time_step:
        # Iterate through each layer
        for ly in range(layers):
            # Forward propagation for the current time step and layer
            if ly == 0:
                h_next, y_pred = rnn_cells[ly].forward(H[ts, ly], X[ts])
            else:
                h_next, y_pred = rnn_cells[ly].forward(H[ts, ly], h_next)
            # Store the next hidden state
            H[ts + 1, ly, :, :] = h_next
        # Store the output for the current time step
        Y.append(y_pred)

    # Convert the list of outputs to numpy array
    Y = np.array(Y)
    return H, Y
