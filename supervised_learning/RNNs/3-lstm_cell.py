#!/usr/bin/env python3
"""
the class LSTMCell that represents an LSTM unit,Creates 
the public instance attributes Wf, Wu, Wc, Wo, Wy, bf,
bu, bc, bo, by that represent 
the weights and biases of the cell

"""
import numpy as np

class LSTMCell:
    """
    Represents a Long Short-Term Memory (LSTM) cell.

    Attributes:
        Wf (numpy.ndarray): Weight matrix for the forget gate.
        Wu (numpy.ndarray): Weight matrix for the update gate.
        Wc (numpy.ndarray): Weight matrix for the intermediate cell state.
        Wo (numpy.ndarray): Weight matrix for the output gate.
        Wy (numpy.ndarray): Weight matrix for the outputs.
        bf (numpy.ndarray): Bias vector for the forget gate.
        bu (numpy.ndarray): Bias vector for the update gate.
        bc (numpy.ndarray): Bias vector for the intermediate cell state.
        bo (numpy.ndarray): Bias vector for the output gate.
        by (numpy.ndarray): Bias vector for the outputs.
    """

    def __init__(self, i, h, o):
        """
        Initializes the LSTMCell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Initialize weight matrices and bias vectors
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def tanh(self, z):
        """
        Hyperbolic tangent activation function.

        Args:
            z (numpy.ndarray): Input to the tanh function.

        Returns:
            numpy.ndarray: Result of applying the tanh function.
        """
        return np.tanh(z)

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

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state.
            c_prev (numpy.ndarray): Previous cell state.
            x_t (numpy.ndarray): Data input for the cell.

        Returns:
            tuple: A tuple containing:
                h_next (numpy.ndarray): Next hidden state.
                c_next (numpy.ndarray): Next cell state.
                y (numpy.ndarray): Output of the cell.
        """
        # Concatenate previous hidden state and input data
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute forget gate
        forget_gate = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        # Compute update gate
        update_gate = self.sigmoid(np.dot(concat, self.Wu) + self.bu)
        # Compute intermediate cell state
        c_cand = self.tanh(np.dot(concat, self.Wc) + self.bc)
        # Compute next cell state
        c_next = forget_gate * c_prev + update_gate * c_cand
        # Compute output gate
        output_gate = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
        # Compute next hidden state
        h_next = output_gate * self.tanh(c_next)
        # Compute output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        
        return h_next, c_next, y

