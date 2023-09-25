#!/usr/bin/env python3
"""
8-EM.py
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform the Expectation-Maximization (EM) algorithm for a Gaussian Mixture Model (GMM).

    Args:
        X (numpy.ndarray): Data points of shape (n_samples, n_features).
        k (int): Number of clusters.
        iterations (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance of the log likelihood, used for early stopping.
        verbose (bool): Determines if information about the algorithm should be printed.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
        - updated_priors (numpy.ndarray): Updated priors for each cluster of shape (k,).
        - updated_means (numpy.ndarray): Updated centroid means for each cluster of shape (k, n_features).
        - updated_covariances (numpy.ndarray): Updated covariance matrices for each cluster of shape (k, n_features, n_features).
        - posterior_probs (numpy.ndarray): Probabilities for each data point in each cluster of shape (k, n_samples).
        - log_likelihood (float): Log likelihood of the model.

    Returns (None, None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n_samples, n_features = X.shape
    priors, cluster_means, cluster_covariances = initialize(X, k)
    prev_log_likelihood = 0

    for iteration in range(iterations + 1):
        posterior_probs, log_likelihood = expectation(
            X, priors, cluster_means, cluster_covariances)

        if iteration != 0:
            priors, cluster_means, cluster_covariances = maximization(
                X, posterior_probs)

        if verbose:
            if iteration % 10 == 0 or iteration == iterations or abs(
                    log_likelihood - prev_log_likelihood) <= tol:
                print(
                    "Log Likelihood after {} iterations: {}".format(
                        iteration, log_likelihood.round(5)))

        if abs(log_likelihood - prev_log_likelihood) <= tol:
            break

        prev_log_likelihood = log_likelihood

    return priors, cluster_means, cluster_covariances, posterior_probs, log_likelihood
