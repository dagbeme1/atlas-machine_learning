#!/usr/bin/env python3

import pandas as pd  # Import the pandas library for data manipulation
from_file = __import__('2-from_file').from_file  # Import the from_file function from another script

# Load the CSV data into a DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Drop the 'Weighted_Price' column
df = df.drop(columns=['Weighted_Price'])

# Forward fill missing values in the 'Close' column
df['Close'] = df['Close'].fillna(method='ffill')

# Fill missing values in 'Volume_(BTC)' and 'Volume_(Currency)' with 0
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

# Fill missing values in 'Open', 'High', and 'Low' with the previous 'Close' value or 0 if not available
df[['Open', 'High', 'Low']] = df[['Open', 'High', 'Low']].apply(lambda x: x.fillna(df['Close'].shift(1).fillna(0)))

# Print the first 10 rows of the DataFrame after cleaning
print(df.head())

# Print the last 10 rows of the DataFrame after cleaning
print(df.tail())
