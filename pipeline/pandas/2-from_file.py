#!/usr/bin/env python3
"""
A function to load data from a file using a specified delimiter
and returns it as a pandas DataFrame.
"""

import pandas as pd  # Import the pandas library for data manipulation

def from_file(filename, delimiter):
    """
    Loads data from a file into a DataFrame.

    Parameters:
        filename (str): The file path to load data from.
        delimiter (str): The column separator/delimiter in the file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    # Read the file with the specified delimiter into a pandas DataFrame
    data_frame = pd.read_csv(filename, delimiter=delimiter)
    
    return data_frame  # Return the resulting DataFrame
