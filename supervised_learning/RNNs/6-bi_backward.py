#!/usr/bin/env python3
"""
the class BidirectionalCell, based on 5-bi_forward.py
public instance method def backward(self, h_next, x_t):
that calculates the hidden
state in the backward direction for one time step

"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN.
    """

    def __init__(self, i, h, o):
        """
        Initializes the BidirectionalCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """
        # Initialize weights and biases for forward and backward directions
        self.Whf = np.random.randn(h + i, h)
        self.Whb = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h * 2, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state.
            x_t (numpy.ndarray): Data input for the cell.

        Returns:
            numpy.ndarray: Next hidden state.
        """
        # Calculate next hidden state in the forward direction
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward
        direction for one time step.

        Args:
            h_next (numpy.ndarray): Next hidden state.
            x_t (numpy.ndarray): Data input for the cell.

        Returns:
            numpy.ndarray: Previous hidden state.
        """
        # Concatenate next hidden state and input data
        concat = np.concatenate((h_next, x_t), axis=1)
        # Calculate previous hidden state in the backward direction
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)
        return h_prev
