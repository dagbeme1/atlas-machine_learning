#!/usr/bin/env python3
"""
Correlation Calculator
"""
import numpy as np


def correlation(C):
    """Calculates a correlation matrix.

    Args:
        C (np.ndarray): covariance of shape (d, d)

    Returns:
        correlation matrix.
    """
    # Check if C is a numpy.ndarray
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Check if C is a square matrix
    if C.shape != C.T.shape:
        raise ValueError("C must be a square matrix")

    # Calculate the square root of the diagonal elements of C
    diagonal = np.sqrt(np.diag(C))

    # Calculate the correlation matrix using broadcasting
    correlation = C / (diagonal[:, None] * diagonal[None, :])

    # Fill the diagonal of the correlation matrix with 1.0
    np.fill_diagonal(correlation, 1.0)

    # Return the calculated correlation matrix
    return correlation
