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
        n_clusters, n_features = C.shape

        # Calculate the squared Euclidean distances between data points and
        # centroids
        centroids_squared_norm = np.sum(C ** 2, axis=1)[:, np.newaxis]
        data_squared_norm = np.sum(X ** 2, axis=1)
        dot_product = np.matmul(C, X.T)
        squared_distances = centroids_squared_norm + data_squared_norm - 2 * dot_product

        # Find the minimum squared distance for each data point
        min_squared_distances = np.amin(squared_distances, axis=0)

        # Calculate the total intra-cluster variance
        total_variance = np.sum(min_squared_distances)

        return total_variance

    except Exception:
        return None
