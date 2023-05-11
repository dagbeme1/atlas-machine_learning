#!/usr/bin/env python3
""" concat matrix """


def cat_matrices2D(mat1, mat2, axis=0):  # defines a function named cat_matrices2D that takes two parameters mat1, mat2
    """ concat matrix based on axis """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2  # block of code checks if the axis parameter is set to 0. If it is, it verifies that the number of columns in mat1 is equal to the number in 2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]  # block of code checks if the axis parameter is set to 1. If it is, it verifies that the number of rows in mat1 is equal to the number of rows in mat2
    else:
        return None  # handles the case when axis is neither 0 nor 1, returning None to indicate an invalid axis.
