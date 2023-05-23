#!/usr/bin/env python3
"""Module with Exponential class"""


class Exponential:

    EULER_NUMBER = 2.7182818285
    
    def __init__(self, data=None, lambtha=1.):
        """
        Initialize an Exponential distribution.

        Args:
            data: List of data to estimate the distribution (default: None).
            lambtha: Expected number of occurrences in a given time frame.

        Raises:
            ValueError: If lambtha is not a positive value.
            TypeError: If data is not a list.
            ValueError: If data does not contain at least two data points.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            self.set_lambtha(data)

    def set_lambtha(self, data):
        """
        Set the lambtha attribute based on the given data.

        Args:
            data: List of data points.

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
        self.lambtha = float(count) / total
