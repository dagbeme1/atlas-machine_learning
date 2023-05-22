#!/usr/bin/env python3
"""Module with Poisson class"""


class Poisson:
    """Class that represents a Poisson distribution"""

    EULER_NUMBER = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            self.set_lambtha(lambtha)
        else:
            self.set_lambtha(self.calculate_lambtha(data))

    def set_lambtha(self, lambtha):
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        self.lambtha = float(lambtha)

    def calculate_lambtha(self, data):
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")
        total = sum(data)
        count = len(data)
        return float(total) / count
