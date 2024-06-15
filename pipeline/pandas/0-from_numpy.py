#!/usr/bin/env python3
"""
a function def from_numpy(array): 
that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd  # Import pandas library for data manipulation

def from_numpy(array):
    """
    Function to create a pandas DataFrame from a NumPy ndarray.

    Parameters:
        array (numpy.ndarray): The input NumPy array.

    Returns:
        pandas.DataFrame: The newly created DataFrame.
    """
    pd.set_option('display.max_columns', None)  # Ensure all columns are displayed in the DataFrame

    # Create a list of column names using ASCII values for uppercase letters
    columns = [chr(i) for i in range(65, 65 + array.shape[1])]

    # Create and return the DataFrame with the specified column names
    return pd.DataFrame(array, columns=columns)
