#!/usr/bin/env python3
"""
7-maximization - a function def maximization(X, g):
that computes the maximization step in the EM algorithm for a GMM
"""
import numpy as np


def maximization(X, g):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n_samples, n_features = X.shape
    n_clusters = g.shape[0]

    if g.shape[1] != n_samples:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), np.ones(n_samples)).all():
        return None, None, None

    updated_priors = np.sum(g, axis=1) / n_samples

    updated_means = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    updated_covariances = np.zeros((n_clusters, n_features, n_features))

    for i in range(n_clusters):
        diff = X - updated_means[i]
        updated_covariances[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    return updated_priors, updated_means, updated_covariances
