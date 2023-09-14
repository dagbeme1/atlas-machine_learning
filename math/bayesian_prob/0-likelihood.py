#!/usr/bin/env python3
"""
Combined Likelihood
"""

import numpy as np

def likelihood(x, n, probability_array):
    """
    Calculate the likelihood of obtaining the data given various hypothetical probabilities
    of developing severe side effects.

    Args:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        probability_array (np.ndarray): Array of hypothetical probabilities.

    Returns:
        np.ndarray: Array containing the likelihood of obtaining the data for each probability.

    Raises:
        ValueError: If n is not a positive integer, if x is not a valid integer, if x > n,
                    or if any value in probability_array is not in [0, 1].
        TypeError: If probability_array is not a 1D numpy.ndarray.
    """

    # Validate the input arguments
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be a non-negative integer")

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(probability_array, np.ndarray) or probability_array.ndim != 1:
        raise TypeError("probability_array must be a 1D numpy.ndarray")

    if not np.all((probability_array >= 0) & (probability_array <= 1)):
        raise ValueError("All values in probability_array must be in the range [0, 1]")

    # Calculate the likelihood for each probability in probability_array
    likelihoods = []
    for p in probability_array:
        comb = np.math.comb(n, x)
        term1 = p ** x
        term2 = (1 - p) ** (n - x)
        likelihood_value = comb * term1 * term2
        likelihoods.append(likelihood_value)

    return np.array(likelihoods)
