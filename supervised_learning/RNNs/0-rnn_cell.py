#!/usr/bin/env python3
"""Class RNNCell"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN.

    Attributes:
        Wh (numpy.ndarray): Weight matrix for the concatenated hidden 
        state and input data.
        bh (numpy.ndarray): Bias vector for the hidden state.
        Wy (numpy.ndarray): Weight matrix for the output.
        by (numpy.ndarray): Bias vector for the output.
    """

    def __init__(self, i, h, o):
        """
        Initialize class constructor

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward propagation vanilla RNN cell

        Args:
            h_prev (numpy.ndarray): Previous hidden state.
            x_t (numpy.ndarray): Data input for the cell.

        Returns:
            tuple: A tuple containing:
                h_next (numpy.ndarray): Next hidden state.
                y (numpy.ndarray): Output of the cell.
        """
        # Concatenate previous hidden state and input data
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state
        h_next = np.tanh(np.dot(concat_input, self.Wh) + self.bh)

        # Calculate output
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1,
                               keepdims=True)  # Softmax activation

        return h_next, y
