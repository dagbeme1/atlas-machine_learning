#!/usr/bin/env python3
""" ridin bareback"""


def mat_mul(mat1, mat2):
    """ multiply two matrix """
    # Check if the matrices can be multiplied
    if len(mat1[0]) != len(mat2):
        return None

    # Create an empty result matrix
    result = [[0] * len(mat2[0]) for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
                                 
    return result
