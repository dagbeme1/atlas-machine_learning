#!/usr/bin/env python3
"""Write a function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Return shape of given list-matrix"""
    shape = []  # empty list which stores dimensions of the matrix
    try:  # a block of code that we want to try running, and if there's an error, we can handle it later.
        while(len(matrix) > 0):  # continues loop as long as length as matrix is greater than zero
            shape.append(len(matrix))  # adds the length of the current dimension to the shape list
            matrix = matrix[0]  # updates the matrix to the first element of the current dimension
    except TypeError:  # something goes wrong and it's a TypeError, we can handle it without the program crashing
        pass
    return shape
