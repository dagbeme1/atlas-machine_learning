#!/usr/bin/env python3

import pandas as pd  # Import the pandas library for data manipulation
from_file = __import__('2-from_file').from_file  # Import the from_file function from another script

# Load the CSV data into a DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Set the 'Timestamp' column as the index of the DataFrame
df = df.set_index('Timestamp')

# Print the last 5 rows of the DataFrame
print(df.tail())
