#!/usr/bin/env python3

"""
This script demonstrates how to:
- Load data from a CSV file into a pandas DataFrame.
- Extract specific columns from the DataFrame.
- Convert the extracted DataFrame to a NumPy array.
- Print the resulting NumPy array.
"""

import pandas as pd  # Import the pandas library for data manipulation

# Import the from_file function from the module '2-from_file'
from_file = __import__('2-from_file').from_file

# Load the CSV file into a DataFrame using the from_file function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Extract the last 10 rows of the "High" and "Close" columns and convert to a NumPy array
A = df[['High', 'Close']].tail(10).to_numpy()

# Print the resulting NumPy array
print(A)
