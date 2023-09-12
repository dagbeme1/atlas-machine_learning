#!/usr/bin/env python3
"""
Advanced Linear Algebra (Minor)
"""


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        list of lists: The minor matrix.
    """
    # Check if matrix is a list of lists
    if not isinstance(
        matrix,
        list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get the size (number of rows) of the matrix
    n = len(matrix)

    # Check if the matrix is square and non-empty
    if n == 0 or len(matrix[0]) != n:
        raise ValueError("matrix must be a non-empty square matrix")

    # Handle the case of a 1x1 matrix
    if n == 1:
        return [[1]]

    # Initialize an empty list for the minor matrix
    minor_mat = []

    # Iterate through the rows of the matrix
    for i in range(n):
        # Initialize an empty list for the current row of the minor matrix
        minor_row = []
        # Iterate through the columns of the matrix
        for j in range(n):
            # Create a submatrix by excluding the current row and column
            submatrix = [row[:j] + row[j + 1:]
                         for row in (matrix[:i] + matrix[i + 1:])]
            # Calculate the determinant using the provided code
            minor_row.append(determinant(submatrix))
        # Append the minor row to the minor matrix
        minor_mat.append(minor_row)

    # Return the minor matrix
    return minor_mat


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        The determinant.
    """
    # Check if the input is a valid list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is empty
    if len(matrix) == 0:
        return 1  # 0x0 matrix, return 1 by convention

    # Check if all elements are lists
    if not all(isinstance(i, list) for i in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get the number of rows and columns in the matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Base cases for 1x1 matrix and empty matrices
    if num_rows == 1 and num_cols == 1:
        return matrix[0][0]
    if num_rows == 1 and num_cols == 0:
        return 1

    # Check if the matrix is square
    if num_rows != num_cols:
        raise ValueError("matrix must be a square matrix")

    # Base case for 2x2 matrix
    if num_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Initialize the determinant
    det = 0

    # Iterate through the column indices
    for col in range(num_cols):
        # Create a submatrix without the first row and the current column
        submatrix = [row[0:col] + row[col + 1:] for row in matrix[1:]]
        # Calculate the cofactor
        cofactor = matrix[0][col] * determinant(submatrix)
        # Add or subtract the cofactor to the determinant with alternating
        # signs
        det += cofactor if col % 2 == 0 else -cofactor

    return det
