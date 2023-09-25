#!/usr/bin/env python3
"""
A function to initialize variables for a Gaussian Mixture Model (GMM)
"""

import numpy as np
# Import kmeans function
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model (GMM).

    Args:
        X (numpy.ndarray): The input dataset with shape (n_samples, n_features).
        k (int): The number of clusters.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        A tuple containing:
            - priors: A numpy.ndarray of shape (k,)
            containing the priors for each cluster, initialized evenly.
            - means: A numpy.ndarray of shape (k, n_features)
            containing the centroid means for each cluster,
            initialized with K-means.
            - covariances: A numpy.ndarray of shape
            (k, n_features, n_features) containing
            the covariance matrices for each cluster, initialized as identity matrices.
            Returns (None, None, None) on failure.
    """
    # Check if X is a numpy array and has 2 dimensions (n_samples, n_features)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # Get the number of samples (n_samples) and the
    # number of features (n_features) from the input dataset X
    n_samples, n_features = X.shape

    # Check if k is a positive integer within the valid range
    if not isinstance(k, int) or k <= 0 or k > n_samples:
        return None, None, None

    # Initialize the "priors" array of shape (k,)
    # containing the priors for each cluster, initialized evenly
    priors = np.tile(1 / k, (k,))

    # Initialize the "means" array of shape (k, n_features)
    # containing the centroid means for each cluster, initialized with K-means
    cluster_means, cluster_assignments = kmeans(X, k)

    # Initialize the "covariances" array of shape (k, n_features, n_features)
    # containing the covariance matrices for each cluster,
    # initialized as identity matrices
    identity_matrix = np.identity(n_features)
    covariances = np.tile(identity_matrix, (k, 1, 1))

    return priors, cluster_means, covariances
