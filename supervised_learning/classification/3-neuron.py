#!/usr/bin/env python3  # Shebang line specifying the interpreter to use

"""Neuron class"""  # Docstring providing a brief description of the code

import numpy as np  # Importing the numpy library for numerical computations

class Neuron:
    """Class that defines a single neuron performing binary classification"""
    # Docstring explaining the purpose of the class

    def __init__(self, nx):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        # Check if nx is an integer, raise an exception if not
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        # Check if nx is a positive integer, raise an exception if not
        self.__W = np.random.randn(1, nx)
        # Initialize the weights with random values
        self.__b = 0  # Initialize the bias to 0
        self.__A = 0  # Initialize the activated output to 0

    @property
    def W(self):
        """W getter"""  # Docstring explaining the purpose of the getter meth
        return self.__W  # Return the value of the private attribute W

    @property
    def b(self):
        """b getter"""  # Docstring explaining the purpose of the getter meth
        return self.__b  # Return the value of the private attribute b

    @property
    def A(self):
        """A getter"""  # Docstring explaining the purpose of the getter meth
        return self.__A  # Return the value of the private attribute A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X: input data

        Returns:
            Activation function - calculated with sigmoid function
        """  # Docstring explaining the purpose of the forward_prop method
        A_prev = np.matmul(self.__W, X) + self.__b
        # Weighted sum of inputs and bias
        self.__A = 1 / (1 + np.exp(-A_prev))
        # Apply sigmoid activation function
        return self.__A  # Return the activated output

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example

        Returns:
            The cost
        """  # Docstring explaining the purpose of the cost method
        m = Y.shape[1]  # Number of training examples
        cost = - (1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                  np.multiply(1 - Y, np.log(1.0000001 - A)))
        # Compute cost using logistic regression formula
        return cost  # Return the computed cost