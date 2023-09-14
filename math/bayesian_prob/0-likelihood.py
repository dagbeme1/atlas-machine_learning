#!/usr/bin/env python3

"""Likelihood"""

import numpy as np

def likelihood(x, n, P):
    """
    Calculate the likelihood of observing the data given various hypothetical probabilities
    of developing severe side effects.

    Args:
        x (int): Number of patients experiencing severe side effects.
        n (int): Total number of patients observed.
        P (np.ndarray): Array of hypothetical probabilities.

    Returns:
        np.ndarray: Array containing the likelihood of observing the data for each probability.

    Raises:
        ValueError: If n is not a positive integer, if x is not a valid integer, if x > n,
                    or if any value in P is not in [0, 1].
        TypeError: If P is not a 1D numpy.ndarray.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the likelihood for each probability in P
    binomial_coefficient = np.math.comb(n, x)
    likelihood_values = binomial_coefficient * (P ** x) * ((1 - P) ** (n - x))

    return likelihood_values

