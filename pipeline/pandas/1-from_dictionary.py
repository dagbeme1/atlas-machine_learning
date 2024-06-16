#!/usr/bin/env python3
"""
Function to create a pandas DataFrame from a dictionary.

Parameters:
    dic (dict): The input dictionary containing data for the DataFrame.

Returns:
    pandas.DataFrame: The newly created DataFrame.
"""

import pandas as pd  # Import pandas library for data manipulation

# Create a dictionary with sample data
dic = {
    "First": [0.0, 0.5, 1.0, 1.5],  # List of values for the 'First' column
    "Second": ["one", "two", "three", "four"]  # List of values for the 'Second' column
}

# Create a DataFrame from the dictionary with specified index labels
df = pd.DataFrame(dic, ["A", "B", "C", "D"])  # 'dic' is the data, ["A", "B", "C", "D"] are the index labels
