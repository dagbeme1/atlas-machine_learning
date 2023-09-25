#!/usr/bin/env python3
"""
A function to initialize variables for a Gaussian Mixture Model (GMM)
"""

import numpy as np
# Import kmeans function
kmeans = __import__('1-kmeans').kmeans

def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): The input dataset with shape (n_samples, n_features) containing the dataset for K-means clustering.
        k (int): A positive integer containing the number of clusters.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        A tuple containing:
            - priors (numpy.ndarray): The priors for each cluster, initialized evenly.
            - centroids (numpy.ndarray): The centroid means for each cluster, initialized with K-means.
            - covariances (numpy.ndarray): The covariance matrices for each cluster, initialized as identity matrices.
            Returns (None, None, None) on failure.
    """
    # Check if X is a numpy array and has 2 dimensions (n_samples, n_features)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # Get the dimensions of the input dataset X
    n_samples, n_features = X.shape

    # Check if k is a positive integer within the valid range
    if not isinstance(k, int) or k <= 0 or k > n_samples:
        return None, None, None

    # Initialize the "priors" array of shape (k,) containing the priors for each cluster, initialized evenly
    priors = np.full(shape=(k,), fill_value=1/k)

    # Initialize the "centroids" array of shape (k, n_features) containing the centroid means for each cluster, initialized with K-means
    centroids, _ = kmeans(X, k)

    # Initialize the "covariances" array of shape (k, n_features, n_features) containing the covariance matrices for each cluster, initialized as identity matrices
    covariance_identity = np.identity(n_features)
    covariances = np.tile(covariance_identity, (k, 1, 1))

    return priors, centroids, covariances

