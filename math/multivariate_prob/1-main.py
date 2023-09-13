#!/usr/bin/env python3

if __name__ == '__main__':
    # This conditional checks if the script is being run as the main program.
    # It ensures that the following code block is only executed when the script
    # is run directly and not when it is imported as a module in another script.

    import numpy as np
    # Import the NumPy library and alias it as 'np'. NumPy is a popular library
    # for numerical operations on arrays and matrices.

    correlation = __import__('1-correlation').correlation
    # Import the 'correlation' function from a module named '1-correlation'.
    # The '__import__' function is used here to dynamically import a module by name.

    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    # Create a NumPy array 'C' representing a 3x3 matrix with specific values.

    Co = correlation(C)
    # Calculate the correlation matrix by calling the 'correlation' function on 'C'.

    print(C)
    # Print the original matrix 'C'.

    print(Co)
    # Print the calculated correlation matrix 'Co'.

