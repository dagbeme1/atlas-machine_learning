#!/usr/bin/env python3
""" adding 2d matrices """


def add_matrices2D(mat1, mat2):
    """ add 2 matrices or return """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):  # checks if the number of rows and number of columns of mat1 and mat2 are not the same
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):  # iterates over the rows and columns of mat1 (or mat2 since they have the same shape) using nested loops
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
