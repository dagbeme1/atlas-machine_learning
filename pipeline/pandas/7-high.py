#!/usr/bin/env python3

import pandas as pd  # Import the pandas library for data manipulation
from_file = __import__('2-from_file').from_file  # Import the from_file function from another script

# Load the CSV data into a DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Sort the DataFrame by 'High' column in descending order
df = df.sort_values('High', ascending=False)

# Print the rows of the sorted DataFrame
print(df.head())
