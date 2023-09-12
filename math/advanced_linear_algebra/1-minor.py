#!/usr/bin/env python3
"""
Advanced Linear Algebra Minor
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
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get the size (number of rows) of the matrix
    n = len(matrix)

    # Check if the matrix is square and non-empty
    if n == 0 or len(matrix[0]) != n:
        raise ValueError("matrix must be a non-empty square matrix")

    # Handle the base case of a 1x1 matrix
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
            submatrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            # Calculate the determinant of the submatrix and append it to the minor row
            minor_row.append(determinant(submatrix))
        # Append the minor row to the minor matrix
        minor_mat.append(minor_row)

    # Return the minor matrix
    return minor_mat

def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        float: The determinant value.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is square (all rows have the same length)
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Get the size (number of rows) of the matrix
    n = len(matrix)

    # Handle the base cases for 1x1 and 2x2 matrices
    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Initialize the determinant value
    det = 0

    # Iterate through the first row (expansion by minors)
    for j in range(n):
        # Calculate the cofactor (product of current element and minor determinant)
        cofactor = matrix[0][j] * determinant([row[:j] + row[j+1:] for row in matrix[1:]])
        # Add or subtract the cofactor to the determinant based on the column index
        det += cofactor if j % 2 == 0 else -cofactor

    # Return the determinant value
    return det
