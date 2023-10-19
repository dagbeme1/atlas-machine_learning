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
    """
    Represents a noiseless 1D Gaussian process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize a Gaussian Process.

        Args:
            X_init (numpy.ndarray): Initial input data points, shape (t, 1).
            Y_init (numpy.ndarray): Initial output data points, shape (t, 1).
            l (float, optional): Length parameter for the kernel.
            Defaults to 1.
            sigma_f (float, optional): Standard deviation of the output.
            Defaults to 1.

        This constructor initializes a 1D Gaussian Process with
        input-output data points.
        The length parameter (l) controls the length scale of the
        kernel, which influences
        the smoothness of functions. The standard deviation (sigma_f)
        represents the noise
        level in the output.

        The class attributes X, Y, and K are also initialized.
        X represents the input data,
        Y represents the output data, and K represents the
        initial covariance kernel matrix.
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
            X1 (numpy.ndarray): Matrix of shape (m, 1) representing
            the input points.
            X2 (numpy.ndarray): Matrix of shape (n, 1) representing
            the input points.

        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n).

        This method computes the covariance kernel matrix based on
        the input points. It uses the
        radial basis function (RBF) kernel to measure the similarity
        between input points in space.
        The hyperparameters, signal variance (sigma_f) and length
        scale (l), are used to control
        the shape of the kernel.

        """

        # Calculate squared distances between inputs
        squared_distances = np.sum(X1 ** 2, axis=1, keepdims=True) + \
            np.sum(X2 ** 2, axis=1) - 2 * np.matmul(X1, X2.T)

        # Compute the covariance kernel matrix (K)
        K = (self.sigma_f ** 2) * \
            np.exp(-0.5 * (1 / (self.l ** 2)) * squared_distances)

        return K

    def predict(self, X_s):
        """
        Predict the mean and standard deviation of points in a
        Gaussian process.

        Args:
            X_s (numpy.ndarray): Points for which the mean and standard
            deviation should be calculated.

        Returns:
            numpy.ndarray: Mean for each point in X_s.
            numpy.ndarray: Standard deviation for each point in X_s.

        This method predicts the mean and standard deviation for
        the specified input points
        using the Gaussian process. It uses the kernel matrix and
        the training data to make
        these predictions. The result is returned as arrays of
        means and standard deviations.

        """

        # Calculate the covariance kernel matrix
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        # Calculate the mean and covariance matrix
        K_inv = np.linalg.inv(K)
        predict_mean = np.matmul(np.matmul(K_s.T, K_inv), self.Y).reshape(-1)
        cov_s = K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)
        sigma = np.diag(cov_s)

        return predict_mean, sigma

    def update(self, X_new, Y_new):
        """
        Update the Gaussian process with new sample points.

        Args:
            X_new (numpy.ndarray): New sample points.
            Y_new (numpy.ndarray): New sample function values.

        This method updates the Gaussian process with new sample
        points and their associated
        function values. It appends these points to the existing
        data and recomputes the kernel
        matrix to incorporate the new information.

        """

        # Append the new sample point and function value
        self.X = np.concatenate((self.X, X_new[..., np.newaxis]), axis=0)
        self.Y = np.concatenate((self.Y, Y_new[..., np.newaxis]), axis=0)

        # Recompute the kernel matrix
        self.K = self.kernel(self.X, self.X)
