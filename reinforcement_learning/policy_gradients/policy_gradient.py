#!/usr/bin/env python3
"""
a function that computes to policy with a weight of a matrix
"""

import numpy as np  # Importing the numpy library

def policy(matrix, weights):
    """
    Function to compute the policy with a given weight of a matrix.

    Parameters:
        matrix (numpy.ndarray): The input matrix.
        weights (numpy.ndarray): The weights applied to the matrix.

    Returns:
        numpy.ndarray: The computed policy.
    """
    combined = np.dot(matrix, weights)  # Compute the dot product of the matrix and weights
    # Compute softmax
    exp_values = np.exp(combined - np.max(combined))  # Subtract max value for numerical stability and apply exponential function
    softmax_result = exp_values / np.sum(exp_values)  # Normalize to get the softmax probabilities
    return softmax_result  # Return the computed policy
