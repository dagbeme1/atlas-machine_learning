#!/usr/bin/env python3
"""
GaussianProcess class representing a noiseless 1D Gaussian process.
Public instance method def update(self, X_new, Y_new):
that updates a Gaussian Process
"""

import numpy as np

class GaussianProcess:
    """
    Class that represents a noiseless 1D Gaussian process.

    Attributes:
        X (numpy.ndarray): Inputs already sampled with the black-box function.
        Y (numpy.ndarray): Outputs of the black-box function for each input.
        l (float): Length parameter for the kernel.
        sigma_f (float): Standard deviation given to the output of the black-box function.
        K (numpy.ndarray): Covariance kernel matrix.

    Methods:
        kernel(X1, X2): Calculate the covariance kernel matrix between two matrices.
        predict(X_s): Predict the mean and standard deviation of points in a Gaussian process.
        update(X_new, Y_new): Update the Gaussian Process with new data points.

    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Constructor for GaussianProcess.

        Args:
            X_init (numpy.ndarray): Inputs already sampled with the black-box function.
            Y_init (numpy.ndarray): Outputs of the black-box function for each input.
            l (float, optional): Length parameter for the kernel. Default is 1.
            sigma_f (float, optional): Standard deviation given to the output of the black-box function. Default is 1.

        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
    """
    Calculate the covariance kernel matrix between two matrices.

    Args:
        X1 (numpy.ndarray): Matrix of shape (m, 1).
        X2 (numpy.ndarray): Matrix of shape (n, 1).

    Returns:
        numpy.ndarray: Covariance kernel matrix of shape (m, n).

    """
    distance_matrix = np.sum(X1 ** 2) + np.sum(X2 ** 2) - 2 * np.dot(X1, X2.T)
    return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * distance_matrix)

    def predict(self, X_s):
        """
        Predict the mean and standard deviation of points in a Gaussian process.

        Args:
            X_s (numpy.ndarray): Points for which mean and standard deviation should be calculated.

        Returns:
            numpy.ndarray: Mean for each point in X_s.
            numpy.ndarray: Standard deviation for each point in X_s.
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        # Mean
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = mu_s.reshape(-1)

        # Variance from covariance
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        var_s = np.diag(cov_s)

        return mu_s, var_s

    def update(self, X_new, Y_new):
        """
        Update the Gaussian Process with new data points.

        Args:
            X_new (numpy.ndarray): New input point as a numpy.ndarray of shape (1,).
            Y_new (numpy.ndarray): New output value as a numpy.ndarray of shape (1,).
        """
        # Update X and Y with the new data points
        self.X = np.append(self.X, X_new)
        self.Y = np.append(self.Y, Y_new)

        # Recalculate the covariance kernel matrix K
        self.K = self.kernel(self.X, self.X)

