#!/usr/bin/env python3
"""
Advance Linear Algebra (Determinant)
"""


def determinant(matrix):
    """function that calculates the determinant of a matrix"""
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get the dimensions of the matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Check if the matrix is square
    if num_rows != num_cols:
        raise ValueError("matrix must be a square matrix")

    # Base case: 0x0 matrix
    if num_rows == [[]]:
        return 1

    # Base case: 1x1 matrix
    if num_rows == 1:
        return matrix[0][0]

    # Initialize the determinant
    det = 0

    # Iterate through the column indices
    for col in range(num_cols):
        # Create a submatrix without the first row and the current column
        submatrix = [row[0:col] + row[col+1:] for row in matrix[1:]]
        # Calculate the cofactor
        cofactor = matrix[0][col] * determinant(submatrix)
        # Add or subtract the cofactor to the determinant
        det += cofactor if col % 2 == 0 else -cofactor

    return det
