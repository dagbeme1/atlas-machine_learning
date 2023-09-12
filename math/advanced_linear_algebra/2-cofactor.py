#!/usr/bin/env python3
"""
Determinant
"""

def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        list of lists: The cofactor matrix.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get the size (number of rows) of the matrix
    n = len(matrix)

    # Check if the matrix is square and non-empty
    if n == 0 or len(matrix[0]) != n:
        raise ValueError("matrix must be a non-empty square matrix")

    # Handle the case of an empty square matrix
    if n == 1 and matrix[0][0] == []:
        raise ValueError("matrix must be a non-empty square matrix")

    # Initialize an empty list for the cofactor matrix
    cofactor_mat = []

    # Iterate through the rows of the matrix
    for i in range(n):
        # Initialize an empty list for the current row of the cofactor matrix
        cofactor_row = []
        for j in range(n):
            # Create a submatrix by excluding the current row and column
            submatrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            # Calculate the determinant of the submatrix
            minor = determinant(submatrix)
            # Calculate the cofactor as (-1)^(i+j) times the minor
            cofactor_value = (-1) ** (i + j) * minor
            cofactor_row.append(cofactor_value)
        # Append the cofactor row to the cofactor matrix
        cofactor_mat.append(cofactor_row)

    # Return the cofactor matrix
    return cofactor_mat
