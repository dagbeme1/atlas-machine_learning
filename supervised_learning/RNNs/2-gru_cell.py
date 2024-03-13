#!/usr/bin/env python3
"""
class GRUCell that represents a gated recurrent unit
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU) cell.

    Attributes:
        Wz (numpy.ndarray): Weight matrix for the update gate.
        Wr (numpy.ndarray): Weight matrix for the reset gate.
        Wh (numpy.ndarray): Weight matrix for the intermediate hidden state.
        Wy (numpy.ndarray): Weight matrix for the output.
        bz (numpy.ndarray): Bias vector for the update gate.
        br (numpy.ndarray): Bias vector for the reset gate.
        bh (numpy.ndarray): Bias vector for the intermediate hidden state.
        by (numpy.ndarray): Bias vector for the output.
    """

    def __init__(self, i, h, o):
        """
        Initializes the GRUCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Initialize weight matrices and bias vectors
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, z):
        """
        Sigmoid activation function.

        Args:
            z (numpy.ndarray): Input to the sigmoid function.

        Returns:
            numpy.ndarray: Result of applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """
        Computes softmax values for each set of scores in x.

        Args:
            z (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Softmax output.
        """
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state.
            x_t (numpy.ndarray): Data input for the cell.

        Returns:
            tuple: A tuple containing:
                h_next (numpy.ndarray): Next hidden state.
                y (numpy.ndarray): Output of the cell.
        """
        # Concatenate previous hidden state and input data
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute update gate
        update_gate = self.sigmoid(np.dot(concat, self.Wz) + self.bz)
        # Compute reset gate
        reset_gate = self.sigmoid(np.dot(concat, self.Wr) + self.br)
        # Compute candidate hidden state
        concat2 = np.concatenate((reset_gate * h_prev, x_t), axis=1)
        cct = np.tanh(np.dot(concat2, self.Wh) + self.bh)
        # Compute next hidden state
        h_next = update_gate * cct + (1 - update_gate) * h_prev
        # Compute output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
