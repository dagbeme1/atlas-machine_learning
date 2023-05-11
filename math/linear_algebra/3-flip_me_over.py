#!/usr/bin/env python3
""" Transpose matrix """


def matrix_transpose(matrix):  # defines a function named matrix_transpose that takes a parameter called matrix
    """ Given matrix perform transpose"""
    new_transpose = []  # store the transpose of the matrix.
    for idx in range(len(matrix[0])):  # iterates over the range of numbers from 0 to the number of columns in the matrix
        current = []  # store the elements of the current columns
        for row in matrix:  # a nested loop that iterates over the range of numbers from 0 to the number of rows in the matrix
            current.append(row[idx])  # extracts the element at the row-th row and idx-th column from the original matrix and appends it. Rows to columns
        new_transpose.append(current)  # appends the row list, which represents a column in the transpose, to the transpose list.
    return new_transpose
