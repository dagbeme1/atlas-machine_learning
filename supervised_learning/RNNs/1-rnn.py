#!/usr/bin/env python3
"""
function def rnn(rnn_cell, X, h_0):
that performs forward propagation for a simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Args:
        rnn_cell (RNNCell): An instance of RNNCell
        used for forward propagation.
        X (numpy.ndarray): The data to be used, of shape (t, m, i).
        h_0 (numpy.ndarray): The initial hidden state, of shape (m, h).

    Returns:
        tuple: A tuple containing:
            H (numpy.ndarray): All hidden states, of shape (t+1, m, h).
            Y (numpy.ndarray): All outputs, of shape (t, m, o).
    """
    T, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((T + 1, m, h))
    H[0] = h_0
    h_next = H[0]
    Y = []

    for t in range(T):
        h_next, y = rnn_cell.forward(h_next, X[t])
        H[t + 1] = h_next
        Y.append(y)

    return H, np.array(Y)
