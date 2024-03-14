#!/usr/bin/env python3
"""
H is a numpy.ndarray of shape (t, m, 2 * h) that contains the 
concatenated hidden states from both directions, 
excluding their initialized states
"""
import numpy as np

class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN.

    Attributes:
        Whf (numpy.ndarray): Weight matrix for the forward hidden states.
        Whb (numpy.ndarray): Weight matrix for the backward hidden states.
        Wy (numpy.ndarray): Weight matrix for the outputs.
        bhf (numpy.ndarray): Bias vector for the forward hidden states.
        bhb (numpy.ndarray): Bias vector for the backward hidden states.
        by (numpy.ndarray): Bias vector for the outputs.
    """

    def __init__(self, i, h, o):
        """
        Initializes the BidirectionalCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Initialize weights and biases for forward and backward directions
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state.
            x_t (numpy.ndarray): Data input for the cell.

        Returns:
            numpy.ndarray: Next hidden state.
        """
        # Concatenate previous hidden state and input data
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        # Calculate next hidden state in the forward direction
        h_next = np.matmul(x_concat, self.Whf) + self.bhf
        h_next = np.tanh(h_next)
        return h_next

    def backward(self, h_next, x_t):
        """
        Performs backward propagation for one time step.

        Args:
            h_next (numpy.ndarray): Next hidden state.
            x_t (numpy.ndarray): Data input for the cell.

        Returns:
            numpy.ndarray: Previous hidden state.
        """
        # Concatenate next hidden state and input data
        x_concat = np.concatenate((h_next, x_t), axis=1)
        # Calculate previous hidden state in the backward direction
        h_back = np.matmul(x_concat, self.Whb) + self.bhb
        h_back = np.tanh(h_back)
        return h_back

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        Args:
            H (numpy.ndarray): Concatenated hidden states from both directions.

        Returns:
            numpy.ndarray: Outputs.
        """
        t, m, h_2 = H.shape
        o = self.by.shape[-1]
        Y = np.zeros((t, m, o))

        for step in range(t):
            Y[step] = np.matmul(H[step], self.Wy) + self.by
            Y[step] = np.exp(Y[step]) / np.sum(np.exp(Y[step]), axis=1, keepdims=True)

        return Y
