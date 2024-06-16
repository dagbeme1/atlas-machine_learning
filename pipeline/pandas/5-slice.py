#!/usr/bin/env python3

"""
This script demonstrates how to:
- Load data from a CSV file into a pandas DataFrame.
- Extract specific columns from the DataFrame.
- Select rows at hourly intervals.
- Print the last 5 rows of the resulting DataFrame.
"""

import pandas as pd  # Import the pandas library for data manipulation

# Import the from_file function from the module '2-from_file'
from_file = __import__('2-from_file').from_file

# Load the CSV file into a DataFrame using the from_file function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Extract the "High", "Low", "Close", and "Volume_(BTC)" columns and select rows at hourly intervals
df = df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60, :]

# Print the last 5 rows of the resulting DataFrame
print(df.tail())
