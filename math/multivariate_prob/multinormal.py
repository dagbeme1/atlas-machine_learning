#!/usr/bin/env python3
"""
multinormal
"""

import numpy as np


class MultiNormal:
    def __init__(self, data):
        """
        Initializes a Multivariate Normal distribution.

        Args:
            data (np.ndarray): The dataset of shape (d, n).

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n is less than 2 (data must contain multiple data points).

        Attributes:
            mean (np.ndarray): The mean of data, shape (d, 1).
            cov (np.ndarray): The covariance matrix of data, shape (d, d).
        """
        # Check if data is a numpy.ndarray and has the correct shape
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Check if data contains multiple data points
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean of data
        self.mean = np.mean(data, axis=1).reshape((-1, 1))

        # Calculate the covariance matrix of data
        n = data.shape[1]
        data_centered = data - self.mean
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)
