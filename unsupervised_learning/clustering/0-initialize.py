#!/usr/bin/env python3
"""
0-initialize.py
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): The dataset with shape (n, d).
        k (int): The number of clusters.

    Returns:
        numpy.ndarray: The initialized centroids for each cluster, or None on failure.
    """

    # Check if X is a numpy array with two dimensions
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    # Get the number of data points (n) and the dimensionality of each data point (d)
    n_samples, n_features = X.shape

    # Check if k is a positive integer within a valid range
    if not isinstance(k, int) or k <= 0 or k > n_samples:
        return None

    # Initialize centroids as an array of zeros
    centroids = np.zeros((k, n_features))

    # Randomly choose k distinct data points as initial centroids
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]

    return centroids
