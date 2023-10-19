#!/usr/bin/env python3
"""
Public instance method def update(self, X_new, Y_new):
that updates a Gaussian Process:
X_new is a numpy.ndarray of shape (1,) that represents
the new sample point
Y_new is a numpy.ndarray of shape (1,) that represents
the new sample function value
Updates the public instance attributes X, Y, and K
"""

import numpy as np


class GaussianProcess:
    """A class representing a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, length_scale=1, signal_stddev=1):
        """
        Constructor for GaussianProcess.

        Args:
            X_init (numpy.ndarray): Inputs already sampled
            with the black-box function.
            Y_init (numpy.ndarray): Outputs of the black-box
            function for each input.
            length_scale (float, optional): Length
            parameter for the kernel. Default is 1.
            signal_stddev (float, optional): Standard deviation given to the
            output of the black-box function. Default is 1.
        """
        self.X = X_init
        self.Y = Y_init
        self.length_scale = length_scale
        self.signal_stddev = signal_stddev
        self.K = self.compute_kernel(X_init, X_init)

    def compute_kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix between two matrices.

        Args:
            X1 (numpy.ndarray): Matrix of shape (m, 1).
            X2 (numpy.ndarray): Matrix of shape (n, 1).

        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n).
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + \
            np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return self.signal_stddev ** 2 * \
            np.exp(-0.5 / self.length_scale ** 2 * sqdist)

    def predict(self, X_s):
        """
        Predict the mean and standard deviation of
        points in a Gaussian process.

        Args:
            X_s (numpy.ndarray): Points for which mean and standard
            deviation should be calculated.

        Returns:
            numpy.ndarray: Mean for each point in X_s.
            numpy.ndarray: Standard deviation for each point in X_s.
        """
        K = self.compute_kernel(self.X, self.X)
        K_s = self.compute_kernel(self.X, X_s)
        K_ss = self.compute_kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        # Calculate the mean
        mean = K_s.T.dot(K_inv).dot(self.Y)[:, 0]

        # Calculate the standard deviation
        stddev = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mean, stddev

    def update(self, X_new, Y_new):
        """
        Update the Gaussian Process with new sample points.

        Args:
            X_new (numpy.ndarray): New sample point.
            Y_new (numpy.ndarray): New sample function value.
        """
        self.X = np.append(self.X, X_new)
        self.X = self.X[:, np.newaxis]

        self.Y = np.append(self.Y, Y_new)
        self.Y = self.Y[:, np.newaxis]

        self.K = self.compute_kernel(self.X, self.X)
