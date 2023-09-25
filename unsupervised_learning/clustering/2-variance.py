#!/usr/bin/env python3
"""
2-variance
"""

import numpy as np


def variance(X, C):
    """
    Calculate the total intra-cluster variance for a data set.

    Args:
        X (numpy.ndarray): The input data set with shape (n, d).
        C (numpy.ndarray): The centroid means for each cluster with shape (k, d).

    Returns:
        float: The total intra-cluster variance, or None on failure.
    """
    try:
        # Calculate the squared Euclidean distances between data points and
        # centroids
        distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

        # Find the minimum distance for each data point
        min_distances = np.min(distances, axis=1)

        # Calculate the total intra-cluster variance
        var = np.sum(min_distances)

        return var

    except Exception:
        return None
