#!/usr/bin/env python3
"""
a function to initialize variables for a Gaussian Mixture Model (GMM)
"""
import numpy as np
# Import kmeans function
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    Args:
        X (numpy.ndarray): The input dataset with shape (n_samples, n_features).
        k (int): The number of clusters.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        A tuple containing:
            - priors: A numpy.ndarray of shape (k,) containing the
            priors for each cluster, initialized evenly.
            - means: A numpy.ndarray of shape (k, n_features) containing the
            centroid means for each cluster, initialized with K-means.
            - covariances: A numpy.ndarray of shape (k, n_features, n_features)
            containing the covariance matrices for each cluster,
            initialized as identity matrices.
            Returns (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # Get the number of data points (n_samples)
    # and the number of features (n_features)
    n_samples, n_features = X.shape

    if not isinstance(k, int) or k <= 0 or k > n_samples:
        return None, None, None

    # Initialize the "priors" array of shape (k,)
    # containing the priors for each cluster
    priors = np.tile(1 / k, (k,))

    # Initialize the "means" array of shape (k, n_features) containing
    # the centroid means for each cluster, initialized with K-means
    means, _ = kmeans(X, k)

    # Initialize the "covariances" array of shape
    # (k, n_features, n_features) containing
    # the covariance matrices for each cluster,
    # initialized as identity matrices
    covariances = np.tile(np.identity(n_features), (k, 1, 1))

    return priors, means, covariances
