#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load data into df1 and df2 from respective CSV files
df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Filter df2 to include only rows where 'Timestamp' is less than or equal to 1417411920
df2 = df2[df2['Timestamp'] <= 1417411920]

# Sort df2 by 'Timestamp' in ascending order
df2 = df2.sort_values(by='Timestamp')

# Set the 'Timestamp' column as the index for df1
df1 = df1.set_index('Timestamp')

# Set the 'Timestamp' column as the index for df2
df2 = df2.set_index('Timestamp')

# Concatenate df2 and df1 with keys 'bitstamp' and 'coinbase'
df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

print(df)
