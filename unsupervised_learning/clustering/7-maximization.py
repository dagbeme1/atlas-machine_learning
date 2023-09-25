#!/usr/bin/env python3
"""
7-maximization - a function def maximization(X, g):
that calculates the maximization step in the EM algorithm for a GMM
"""
import numpy as np

def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a Gaussian Mixture Model (GMM).

    :param X: numpy.ndarray of shape (n_samples, n_features)
        containing the data set.
    :param g: numpy.ndarray of shape (n_clusters, n_samples)
        containing the posterior probabilities for each data point in each cluster.
    :return: updated_priors, updated_means, updated_covariances, or None, None, None on failure.
        - updated_priors: numpy.ndarray of shape (n_clusters,)
            containing the updated priors for each cluster.
        - updated_means: numpy.ndarray of shape (n_clusters, n_features)
            containing the updated centroid means for each cluster.
        - updated_covariances: numpy.ndarray of shape (n_clusters, n_features, n_features)
            containing the updated covariance matrices for each cluster.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    if X.shape[0] != g.shape[1]:
        return None, None, None

    n_samples, n_features = X.shape
    n_clusters = g.shape[0]


    if not np.isclose(np.sum(g, axis=0), np.ones(n_samples)).all():
        return None, None, None


    updated_priors = np.zeros(n_clusters)
    updated_means = np.zeros((n_clusters, n_features))
    updated_covariances = np.zeros((n_clusters, n_features, n_features))


    for i in range(n_clusters):
        updated_priors[i] = np.sum(g[i]) / n_samples

    for i in range(n_clusters):
        cluster_weights = g[i]
        total_weight = np.sum(cluster_weights)

        updated_means[i] = np.sum(
            X * cluster_weights.reshape(-1, 1), axis=0) / total_weight

    for i in range(n_clusters):
        cluster_weights = g[i]
        total_weight = np.sum(cluster_weights)

        diff = X - updated_means[i]
        updated_covariances[i] = np.dot(
            (diff.T * cluster_weights), diff) / total_weight

    return updated_priors, updated_means, updated_covariances
