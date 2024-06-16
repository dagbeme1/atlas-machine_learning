#!/usr/bin/env python3

"""
This script demonstrates how to:
- Load data from a CSV file into a pandas DataFrame.
- Rename a column in the DataFrame.
- Convert a timestamp column from Unix time to a datetime object.
- Select specific columns from the DataFrame.
- Print the last few rows of the DataFrame.
"""

import pandas as pd  # Import the pandas library for data manipulation

# Import the from_file function from the module '2-from_file'
from_file = __import__('2-from_file').from_file

# Load the CSV file into a DataFrame using the from_file function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Rename the "Timestamp" column to "Datetime"
df = df.rename(columns={"Timestamp": "Datetime"})

# Convert the "Datetime" column from Unix time to a datetime object
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# Select only the "Datetime" and "Close" columns from the DataFrame
df = df[['Datetime', 'Close']]

# Print the last 5 rows of the DataFrame
print(df.tail())
