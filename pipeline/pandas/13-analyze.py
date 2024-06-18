#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load data from CSV file into DataFrame using custom function 'from_file'
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Calculate descriptive statistics for numerical columns (excluding 'Timestamp')
stats = df[df.columns[df.columns != 'Timestamp']].describe()

# Print the calculated statistics
print(stats)
