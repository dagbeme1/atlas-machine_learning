#!/usr/bin/env python3
"""Module with Exponential class"""


class Exponential:
    """Class that represents an exponential distribution"""

    EULER_NUMBER = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
        # checks for the case data
        if data is not None:
            # checks if data is a list
            if not isinstance(data, list):
                raise TypeError('data must be a list')

            # checks for data containing at least 2 values
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

            # calculating for lambtha value
            self.lambtha = 1/(sum(data)/len(data))
        else:
            # handles case when data is not provided
            # checking if the lambtha is a positivee value
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')

            # Assigning the lambtha value as a float
            self.lambtha = float(lambtha)
