#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load data into df1 and df2 from respective CSV files
df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Sort df1 and df2 by 'Timestamp' in ascending order and set 'Timestamp' as index
df1 = df1.sort_values(by='Timestamp').set_index('Timestamp')
df2 = df2.sort_values(by='Timestamp').set_index('Timestamp')

# Filter df1 and df2 based on Timestamp range
df1 = df1.loc["1417411980":"1417417980"]
df2 = df2.loc["1417411980":"1417417980"]

# Concatenate df1 and df2 with keys 'bitstamp' and 'coinbase', swap levels and sort index
df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase']).swaplevel(i=0, j=1, axis=0).sort_index()

# Print the concatenated and sorted DataFrame
print(df)
