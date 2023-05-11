#!/usr/bin/env python3
""" Transpose matrix """


def matrix_transpose(matrix):
    """ Given matrix perform transpose"""
    new_transpose = []
    for idx in range(len(matrix[0])):
        current = []
        for row in matrix:
            current.append(row[idx])
        new_transpose.append(current)
    return new_transpose
