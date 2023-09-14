#!/usr/bin/env python3

"""Intersection"""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining this data with various hypothetical probabilities.

    Args:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (np.ndarray): Array of hypothetical probabilities.
        Pr (np.ndarray): Array of prior beliefs of P.

    Returns:
        np.ndarray: Array containing the intersection of obtaining x and n with each probability in P.

    Raises:
        ValueError: If n is not a positive integer, if x is not a valid integer, if x > n,
                    if any value in P or Pr is not in [0, 1], or if Pr does not sum to 1.
        TypeError: If P or Pr is not a 1D numpy.ndarray or if Pr does not have the same shape as P.
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

    # Check if P is a 1D numpy array
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if Pr is a 1D numpy array with the same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise ValueError("Pr must be a numpy.ndarray with the same shape as P")

    # Check if values in P are within the range [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError(f"All values in P must be in the range [0, 1]")

    # Check if values in Pr are within the range [0, 1]
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError(f"All values in Pr must be in the range [0, 1]")

    # Check if the sum of values in Pr is approximately equal to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the intersection of obtaining x and n with each probability in
    # P
    intersection_values = likelihood(x, n, P) * Pr

    return intersection_values
