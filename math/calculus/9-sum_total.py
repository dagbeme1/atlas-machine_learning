#!/usr/bin/env python3
""" sum squared """

def summation_i_squared(n):
    if not isinstance(n, int) or n <= 0:
        return None

    sum_of_squares = sum(i ** 2 for i in range(1, n+1))
    return sum_of_squares
