#!/usr/bin/env python3
"""
2-variance
"""

import numpy as np

def variance(X, C):
    """
    Calculate the total intra-cluster variance for a given data set.

    Args:
        X (numpy.ndarray): The dataset with shape (n_samples, n_features).
        C (numpy.ndarray): The centroid means for each cluster with shape (n_clusters, n_features).

    Returns:
        float: The total intra-cluster variance, or None on failure.
    """
    try:
        n_samples, n_features = X.shape
        n_clusters, _ = C.shape

        # Calculate the squared Euclidean distances between data points and centroids
        X_broadcasted = X[:, np.newaxis, :]
        squared_distances = np.sum((X_broadcasted - C) ** 2, axis=2)

        # Find the minimum squared distance for each data point
        min_squared_distances = np.min(squared_distances, axis=1)

        # Calculate the total intra-cluster variance
        total_variance = np.sum(min_squared_distances)

        return total_variance

    except Exception:
        return None
