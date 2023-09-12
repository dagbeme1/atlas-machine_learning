def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        The determinant.
    """
    # Check if the matrix is square
    if matrix and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # Get the number of rows in the matrix
    num_rows = len(matrix)

    # Base case: 0x0 matrix
    if num_rows == 0:
        return 1

    # Base case: 1x1 matrix
    if num_rows == 1:
        return matrix[0][0]

    # Handle base cases for 1x1 and 2x2 matrices
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Initialize the determinant value
    det = 0
    for j in range(len(matrix[0])):
        # Create a submatrix without the first row and the current column
        omited_matrix = minor_m(matrix, 0, j)
        # Calculate the determinant using recursive calls
        det += matrix[0][j] * ((-1) ** j) * determinant(omited_matrix)

    return det
