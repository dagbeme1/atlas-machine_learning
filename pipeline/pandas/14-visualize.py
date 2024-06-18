#!/usr/bin/env python3

from datetime import date  # Importing the 'date' class from the datetime module
import matplotlib.pyplot as plt  # Importing matplotlib's pyplot module with alias 'plt'
import pandas as pd  # Importing pandas library with alias 'pd'
from_file = __import__('2-from_file').from_file  # Importing the 'from_file' function from module '2-from_file'

# Using the from_file function to read data from CSV into DataFrame 'df'
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Rename 'Timestamp' column to 'Date' and convert to datetime
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')  # Converting 'Timestamp' to datetime and assigning it to 'Date'
df = df.drop(columns=['Timestamp'])  # Dropping the original 'Timestamp' column

# Filter data from '2017-01-01' onwards based on 'Date'
df = df[df['Date'] >= '2017-01-01']

# Drop 'Weighted_Price' column from the DataFrame
df = df.drop(columns=['Weighted_Price'])

# Set 'Date' column as the index of the DataFrame
df = df.set_index('Date')

# Fill missing values in 'Close', 'Volume_(BTC)', and 'Volume_(Currency)' columns
df['Close'].fillna(method='ffill', inplace=True)  # Forward fill missing values in 'Close'
df['Volume_(BTC)'].fillna(value=0, inplace=True)  # Fill missing values in 'Volume_(BTC)' with 0
df['Volume_(Currency)'].fillna(value=0, inplace=True)  # Fill missing values in 'Volume_(Currency)' with 0

# Fill missing values in 'Open', 'High', 'Low' using 'Close' shifted by 1 day
df['Open'] = df['Open'].fillna(df['Close'].shift(1, fill_value=0))  # Fill missing 'Open' values with previous day's 'Close'
df['High'] = df['High'].fillna(df['Close'].shift(1, fill_value=0))  # Fill missing 'High' values with previous day's 'Close'
df['Low'] = df['Low'].fillna(df['Close'].shift(1, fill_value=0))  # Fill missing 'Low' values with previous day's 'Close'

# Resample data to daily frequency and aggregate using specified functions
df_sample = df.resample('D').agg({
    'Open': 'first',  # First value of 'Open' in the day
    'High': 'max',    # Maximum value of 'High' in the day
    'Low': 'min',     # Minimum value of 'Low' in the day
    'Close': 'last',  # Last value of 'Close' in the day
    'Volume_(BTC)': 'sum',         # Sum of 'Volume_(BTC)' in the day
    'Volume_(Currency)': 'sum'     # Sum of 'Volume_(Currency)' in the day
})

# Plot the resampled DataFrame using matplotlib
df_sample.plot()  # Plotting the resampled DataFrame
plt.show()  # Displaying the plot using matplotlib.pyplot
