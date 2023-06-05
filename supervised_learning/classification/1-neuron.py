#!/usr/bin/env python3  # Shebang line specifying the interpreter
"""Neuron class"""  # Docstring describing the purpose of the code
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):  # Class constructor
        """Class constructor"""
        if type(nx) is not int:  # Check if nx is not an integer
            raise TypeError('nx must be an integer')  # Raise TypeError
        if nx < 1:  # Check if nx is less than 1
            raise ValueError('nx must be a positive integer')  # Raise ValueError
        self.__W = np.random.randn(1, nx)  # Initialize weights with random values
        self.__b = 0  # Initialize bias to 0
        self.__A = 0  # Initialize activated output to 0

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
