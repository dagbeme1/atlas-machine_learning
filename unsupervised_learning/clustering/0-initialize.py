#!/usr/bin/env python3
"""
0-initialize.py
"""

import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): The input dataset with shape (n_samples, n_features).
        k (int): The number of clusters.

    Returns:
        numpy.ndarray: The initialized centroids for each cluster.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    n_samples, n_features = X.shape

    if not isinstance(k, int) or k <= 0 or k > n_samples:
        return None

    centroids = np.random.uniform(np.min(X, axis=0),
            np.max(X, axis=0), (k, n_features))

    return centroids

