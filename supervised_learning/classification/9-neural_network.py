#!/usr/bin/env python3
"""Neural Network class"""
import numpy as np


class NeuralNetwork:
    """Class that defines a neural network with one hidden performing
    binary classification
    """

    def __init__(self, nx, nodes):
        """Class constructor"""
        # Check the type and value of nx
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        # Check the type and value of nodes
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        # Initialize the private attributes with random values
        # Weight matrix for the hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))  # Bias vector for the hidden layer
        self.__A1 = 0  # Activated output of the hidden layer
        # Weight matrix for the output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0  # Bias for the output layer
        self.__A2 = 0  # Activated output of the neural network

    # Getters for the private attributes
    @property
    def W1(self):
        """W1 getter"""
        return self.__W1

    @property
    def b1(self):
        """b1 getter"""
        return self.__b1

    @property
    def A1(self):
        """A1 getter"""
        return self.__A1

    @property
    def W2(self):
        """W2 getter"""
        return self.__W2

    @property
    def b2(self):
        """b2 getter"""
        return self.__b2

    @property
    def A2(self):
        """A2 getter"""
        return self.__A2
