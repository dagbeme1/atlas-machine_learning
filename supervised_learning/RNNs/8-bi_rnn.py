#!/usr/bin/env python3
"""
the function def bi_rnn(bi_cell, X, h_0, h_t): that performs forward
propagation for a bidirectional RNN
"""
import numpy as np

def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Args:
        bi_cell (BidirectionalCell): Instance of BidirectionalCell
        for forward propagation.
        X (numpy.ndarray): Data input, shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden state in
        the forward direction, shape (m, h).
        h_t (numpy.ndarray): Initial hidden state in
        the backward direction, shape (m, h).

    Returns:
        H (numpy.ndarray): Concatenated hidden states, shape (t, m, 2 * h).
        Y (numpy.ndarray): Outputs, shape (t, m, o).
    """
    t, m, i = X.shape
    _, h = h_0.shape

    H_f = np.zeros((t + 1, m, h))
    H_b = np.zeros((t + 1, m, h))
    H_f[0] = h_0
    H_b[t] = h_t

    # Forward pass
    for ti in range(t):
        H_f[ti + 1] = bi_cell.forward(H_f[ti], X[ti])
        H_b[t - ti - 1] = bi_cell.backward(H_b[t - ti], X[t - ti - 1])

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_f[1:], H_b[:t]), axis=-1)

    # Compute outputs
    Y = bi_cell.output(H)

    return H, Y
