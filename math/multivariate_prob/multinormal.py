#!/usr/bin/env python3
"""
update multinormal
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.

    Attributes:
        mean (np.ndarray): The mean vector of the dataset, shape (d, 1).
        cov (np.ndarray): The covariance matrix of the dataset, shape (d, d).
    """

    def __init__(self, data):
        """
        Initializes a Multivariate Normal distribution.

        Args:
            data (np.ndarray): The dataset of shape (d, n).

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n is less than 2
            (data must contain multiple data points).

        Attributes:
            mean (np.ndarray): The mean vector of the dataset, shape (d, 1).
            cov (np.ndarray): The covariance matrix of the dataset,
            shape (d, d).
        """
        # Check if data is a numpy.ndarray and has the correct shape
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Check if data contains multiple data points
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean vector of data
        self.mean = np.mean(data, axis=1).reshape((-1, 1))

        # Calculate the covariance matrix of data
        n = data.shape[1]
        data_centered = data - self.mean
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the Probability Density Function (PDF) at a data point.

        Args:
            x (np.ndarray): The data point of shape (d, 1).

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x is not of shape (d, 1).

        Returns:
            float: The value of the PDF at the given data point.
        """
        # Check if x is a numpy.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        # Check if x has the correct shape
        if x.shape != self.mean.shape:
            raise ValueError(f"x must have the shape {self.mean.shape}")

        # Calculate the PDF value using the Multivariate Normal PDF formula
        d = self.mean.shape[0]
        x_minus_mean = x - self.mean
        exponent = -0.5 * \
            np.matmul(x_minus_mean.T, np.matmul(np.linalg.inv(self.cov),
                                                x_minus_mean))
        denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        pdf_value = np.exp(exponent) / denominator

        return pdf_value
