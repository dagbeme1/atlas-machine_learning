#!/usr/bin/env python3
"""
0-initialize.py
"""

import numpy as np


def initialize(X, k, seed=None):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): The input dataset with shape (n_samples, n_features).
        k (int): The number of clusters.
        seed (int): Random seed for reproducibility.

    Returns:
        numpy.ndarray: The initialized centroids for each cluster.
    """
    if seed is not None:
        np.random.seed(seed)  # Set the random seed

    # Check if X is a numpy array with two dimensions
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    # Get the number of data points (n_samples) and the dimensionality of each
    # data point (n_features)
    n_samples, n_features = X.shape

    # Check if k is a positive integer within a valid range
    if not isinstance(k, int) or k <= 0 or k > n_samples:
        return None

    # Compute the minimum and maximum values for each feature
    min_val = np.amin(X, axis=0)
    max_val = np.amax(X, axis=0)

    # Initialize centroids using random values within the data range
    centroids = np.random.uniform(min_val, max_val, (k, n_features))

    return centroids
