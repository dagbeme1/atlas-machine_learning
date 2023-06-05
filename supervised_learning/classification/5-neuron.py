#!/usr/bin/env python3

"""Neuron class"""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Class constructor

        Args:
            nx: number of features in the input data
        """
        # Check if nx is an integer
        if type(nx) is not int:
            raise TypeError('nx must be an integer')

        # Check if nx is a positive integer
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        # Initialize the weights (__W) with random values of shape (1, nx)
        self.__W = np.random.randn(1, nx)

        # Initialize the bias (__b) to 0
        self.__b = 0

        # Initialize the activation (__A) to 0
        self.__A = 0

    @property
    def W(self):
        """W getter"""
        return self.__W

    @property
    def b(self):
        """b getter"""
        return self.__b

    @property
    def A(self):
        """A getter"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: input data

        Returns:
            Activation function - calculated with sigmoid function
        """
        # Calculate the weighted sum of inputs and add the bias
        A_prev = np.matmul(self.__W, X) + self.__b

        # Apply the sigmoid activation function
        self.__A = 1 / (1 + np.exp(-A_prev))

        # Return the activation
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example

        Returns:
            The cost
        """
        # Get the number of examples in the dataset
        m = Y.shape[1]

        # Calculate the cost using the logistic regression formula
        cost = - (1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                  np.multiply(1 - Y, np.log(1.0000001 - A)))

        # Return the cost
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data

        Returns:
            The neuron's prediction and the cost of the network
        """
        # Perform forward propagation to calculate the activation
        self.forward_prop(X)

        # Use a threshold of 0.5 to classify the predictions
        predictions = np.where(self.A <= 0.5, 0, 1)

        # Calculate the cost using the predicted values
        cost = self.cost(Y, self.A)

        # Return the predictions and the cost
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example
            alpha: learning rate
        """
        # Get the number of examples in the dataset
        m = Y.shape[1]

        # Calculate the difference between the activated output
        d_ay = A - Y

        # Calculate the gradient of the weights
        gradient = np.matmul(d_ay, X.T) / m

        # Calculate the gradient of the bias
        db = np.sum(d_ay) / m

        # Update the weights and bias using the gradient descent update rule
        self.__W -= gradient * alpha
        self.__b -= db * alpha
