#!/usr/bin/env python3
"""
Advanced Linear Algebra Determinant
"""

def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        The determinant.
    """
    # Check if the input is a valid list of lists
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if all elements are lists
    if all(isinstance(i, list) for i in matrix) is False:
        raise TypeError("matrix must be a list of lists")

    # Get the number of rows and columns in the matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Base case: 0x0 matrix
    if num_rows == 0 and num_cols == 0:
        return 1  # 0x0 matrix, return 1 by convention

    # Base case: 1x1 matrix
    if num_rows == 1 and num_cols == 1:
        return matrix[0][0]

    # Initialize the determinant
    det = 0

    # Choose the method based on the size of the matrix
    if num_rows == num_cols:
        # Square matrix
        if num_rows == 2:
            # Handle base case for 2x2 square matrix
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        else:
            # Iterate through the column indices
            for col in range(num_cols):
                # Create a submatrix without the first row and the current column
                submatrix = [row[0:col] + row[col+1:] for row in matrix[1:]]
                # Calculate the cofactor
                cofactor = matrix[0][col] * determinant(submatrix)
                # Add or subtract the cofactor to the determinant
                det += cofactor if col % 2 == 0 else -cofactor
    else:
        # Non-square matrix
        raise ValueError("matrix must be a square matrix")

    return det
