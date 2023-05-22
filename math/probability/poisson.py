#!/usr/bin/env python3
"""Module with Poisson class"""


class Poisson:
    """Class that represents a Poisson distribution"""

    EULER_NUMBER = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson distribution.

        Args:
            data: List of data to estimate the distribution (default: None).
            lambtha: Expected number of occurrences in a given time frame.

        Raises:
                ValueError: If lambtha is not a positive value.
                TypeError: If data is not a list.
                ValueError: If data does not contain at least two data points.
        """
        if data is None:
            self.set_lambtha(lambtha)
        else:
            self.set_lambtha(self.calculate_lambtha(data))

    def set_lambtha(self, lambtha):
        """
        Set the lambtha attribute.

        Args:
            lambtha: Expected number of occurrences in a given time frame.

        Raises:
            ValueError: If lambtha is not a positive value.
        """
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        self.lambtha = float(lambtha)

    def calculate_lambtha(self, data):
        """
        Calculate the lambtha value based on the given data.

        Args:
            data: List of data points.

        Returns:
                float: The calculated lambtha value.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data does not contain at least two data points.
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")
        total = sum(data)
        count = len(data)
        return float(total) / count

    def pmf(self, k):
        """
        Calculate the value of the PMF (Probability Mass Function)

        Args:
            k: Number of successes.

        Returns:
            float: The PMF value for k.

        """
        k = int(k)
        if k < 0:
            return 0
        return (self.lambtha ** k) * (2.71828 ** -self.lambtha) 
    / self.factorial(k)

    def factorial(self, n):
        """
        Calculate the factorial of a number.

        Args:
            n: The number to calculate the factorial for.

        Returns:
            int: The factorial value of n.

        """
        if n == 0:
            return 1
        else:
            return n * self.factorial(n - 1)
