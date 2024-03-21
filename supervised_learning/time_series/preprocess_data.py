#!/usr/bin/env python3
"""
Bitcoin Price Data Preprocessing
"""

import pandas as pd

def preprocess_data(file_path):
    """
    Preprocess Bitcoin price data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed training data.
        pd.DataFrame: Preprocessed validation data.
        pd.DataFrame: Preprocessed test data.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Convert 'Timestamp' column to datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')

    # Resample data to hourly frequency and forward-fill missing values
    data.set_index('Timestamp', inplace=True)
    data = data.resample('1H').ffill()

    # Filter data for the year 2017 and onwards
    data = data[data.index.year >= 2017]

    # Split data into train, validation, and test sets (70%, 20%, 10%)
    train_size = int(0.7 * len(data))
    val_size = int(0.2 * len(data))
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    # Normalize data using z-score standardization
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    # Save preprocessed data to CSV files
    train_data.to_csv('train_data.csv')
    val_data.to_csv('val_data.csv')
    test_data.to_csv('test_data.csv')

    return train_data, val_data, test_data

if __name__ == "__main__":
    file_path = "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
    preprocess_data(file_path)

