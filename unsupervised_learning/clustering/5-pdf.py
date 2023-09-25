#!/usr/bin/env python3
"""
6-pdf - a function def pdf(X, m, S): that calculates the probability
density function of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function (PDF) of a Gaussian distribution.

    Args:
        X (numpy.ndarray): Data points of shape (n_samples, n_features).
        m (numpy.ndarray): Mean of the distribution of shape (n_features,).
        S (numpy.ndarray): Covariance matrix of shape (n_features, n_features).

    Returns:
        numpy.ndarray: PDF values for each data point of shape (n_samples,).
        Returns None on failure.
    """
    # Check if X is a NumPy array and has 2 dimensions (n_samples, n_features)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    # Check if m is a NumPy array and has 1 dimension (n_features,)
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    # Check if S is a NumPy array and has 2 dimensions (n_features, n_features)
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    # Check if the dimensions of X, m, and S match for computations
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    # Check if S is a square matrix (symmetric)
    if S.shape[0] != S.shape[1]:
        return None

    # Calculate the number of samples and number of features from X
    n_samples, n_features = X.shape

    # Compute the inverse of the covariance matrix S
    inverse_covariance = np.linalg.inv(S)
    # Compute the determinant of the covariance matrix S
    determinant_covariance = np.linalg.det(S)

    # Calculate the denominator for the PDF formula
    denominator = np.sqrt(((2 * np.pi) ** n_features) * determinant_covariance)

    # Reshape m to match dimensions with X
    m_reshaped = m.reshape(-1, 1)

    # Compute the difference between data points and the mean without using
    # loops
    difference = X.T - m_reshaped

    # Compute the intermediate matrix by multiplying with the inverse
    # covariance
    intermediate_matrix = np.matmul(inverse_covariance, difference)

    # Calculate the squared exponents part of the PDF formula without loops
    squared_exponents = np.sum(difference * intermediate_matrix, axis=0)
    # Calculate the exponents part of the PDF formula
    exponents = -0.5 * squared_exponents

    # Calculate the PDF values without using loops
    pdf_values = np.exp(exponents) / denominator

    # Ensure that PDF values have a minimum value of 1e-300
    pdf_values = np.maximum(pdf_values, 1e-300)

    # Return the calculated PDF values
    return pdf_values
