#!/usr/bin/env python3
"""
update multinormal
"""

# Import the NumPy library for numerical operations.
import numpy as np

# Define a class named MultiNormal for representing a Multivariate Normal
# distribution.


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
            mean (np.ndarray): The mean vector of the dataset, shape (d, 1).
            cov (np.ndarray): The covariance matrix of the dataset, shape (d, d).
        """
        # Validate the input data.
        self._validate_input(data)

        # Calculate the mean vector and covariance matrix.
        self.mean = np.mean(data, axis=1).reshape((-1, 1))
        self.cov = self._calculate_covariance(data)

    def _validate_input(self, data):
        """
        Validate the input data.

        Args:
            data (np.ndarray): The dataset to be validated.

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n is less than 2 (data must contain multiple data points).
        """
        # Check if data is a 2D numpy array.
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Check if data contains multiple data points.
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

    def _calculate_covariance(self, data):
        """
        Calculate the covariance matrix of the input data.

        Args:
            data (np.ndarray): The dataset for which the covariance matrix is calculated.

        Returns:
            np.ndarray: The calculated covariance matrix.
        """
        # Calculate the covariance matrix.
        n = data.shape[1]
        data_centered = data - self.mean
        return np.matmul(data_centered, data_centered.T) / (n - 1)

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
        # Validate the input data point.
        self._validate_data_point(x)

        # Calculate the PDF value using the Multivariate Normal PDF formula.
        d = self.mean.shape[0]
        x_minus_mean = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)
        exponent = -0.5 * np.matmul(x_minus_mean.T,
                                    np.matmul(cov_inv, x_minus_mean))
        denominator = np.sqrt((2 * np.pi) ** d * cov_det)
        pdf_value = np.exp(exponent) / denominator
        return pdf_value.flatten()[0]

    def _validate_data_point(self, x):
        """
        Validate the input data point.

        Args:
            x (np.ndarray): The data point to be validated.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x does not have the correct shape.
        """
        # Check if x is a numpy.ndarray.
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        # Check if x has the correct shape.
        if x.shape != self.mean.shape:
            raise ValueError(f"x must have the shape {self.mean.shape}")
