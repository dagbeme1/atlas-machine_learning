#!/usr/bin/env python3
"""Module with Normal class"""


class Normal:
    """Class that represents a normal distribution"""

    EULER_NUMBER = 2.7182818285
    PI = 3.1415926536

    @staticmethod
    def get_stddev(data, mean):
        """Calculates Standard Deviation with a given data and mean"""
        summation = 0
        for number in data:
            summation += (number - mean) ** 2
        return (summation / len(data)) ** (1 / 2)

    @classmethod
    def get_erf(cls, x):
        """Calculates the Error Function (erf) for a given value x"""
        seq = (x - ((x ** 3) / 3)
                + ((x ** 5) / 10)
                - ((x ** 7) / 42)
                + ((x ** 9) / 216))
        erf = (2 / (cls.PI ** (1 / 2))) * seq
        return erf

    def __init__(self, data=None, mean=0, stddev=1.):
        """Class constructor"""

        # Handling case when data is provided
        if data is not None:
            # Checking if data is a list
            if not isinstance(data, list):
                raise TypeError('data must be a list')

            # Checking if data contains at least two values
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

            # Calculating the mean and standard deviation based on data
            self.mean = sum(data) / len(data)
            self.stddev = self.get_stddev(data, self.mean)
        else:
            # Handling case when data is not provided
            # Checking if stddev is a positive value
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')

            # Assigning the mean and standard deviation values as floats
            self.mean = float(mean)
            self.stddev = float(stddev)
