#!/usr/bin/env python3

"""Intersection"""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining this data given various
    hypothetical probabilities of developing severe side effects.

    Args:
        x (int): Number of patients experiencing severe side effects.
        n (int): Total number of patients observed.
        P (np.ndarray): Array of hypothetical probabilities.

    Returns:
        np.ndarray: Array containing the intersection of obtaining
        the data for each probability.

    Raises:
        ValueError: If n is not a positive integer,
        if x is not a valid integer, if x > n, or if any value in P or Pr is not in [0, 1],
        or if Pr does not sum to 1.
    """
    # Check if n is a positive integer
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    # Check if x is greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy array in the range [0, 1]
    if not isinstance(
        P, np.ndarray) or P.ndim != 1 or np.any(
        (P < 0) | (
            P > 1)):
        raise ValueError(
            "P must be a 1D numpy.ndarray with values in the range [0, 1]")

    # Check if Pr is a 1D numpy array with the same shape as P and in the
    # range [0, 1]
    if not isinstance(
        Pr,
        np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape or np.any(
        (Pr < 0) | (
            Pr > 1)):
        raise ValueError(
            "Pr must be a 1D numpy.ndarray with the same shape as P and values in the range [0, 1]")

    # Check if the sum of values in Pr is approximately equal to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute the binomial coefficient manually
    binomial_coefficient = (np.math.factorial(
        n) / (np.math.factorial(x) * np.math.factorial(n - x)))

    # Calculate the likelihood for each probability in P
    intersection_values = binomial_coefficient * \
        (P ** x) * ((1 - P) ** (n - x)) * Pr

    return intersection_values
