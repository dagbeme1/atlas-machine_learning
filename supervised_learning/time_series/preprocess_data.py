#!/usr/bin/env python3:wq


import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define paths to zipped files
bitstamp_zip_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv.zip'
coinbase_zip_path = 'data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip'

# Function to extract zip files
def extract_zip(zip_file, extract_dir):
    """
    Extracts a zip file to a specified directory.

    Args:
    zip_file (str): Path to the zip file.
    extract_dir (str): Directory where the zip file will be extracted.
    """
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Extract zip files
extract_zip(bitstamp_zip_path, 'data')
extract_zip(coinbase_zip_path, 'data')

# Function to load raw data from CSV file
def load_data(file_path):
    """
    Load raw cryptocurrency data from a CSV file.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Loaded raw data.
    """
    return pd.read_csv(file_path)

# Load raw data
bitstamp_raw_data = load_data('data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
coinbase_raw_data = load_data('data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

# Function to preprocess data
def preprocess_data(df):
    """
    Preprocess raw cryptocurrency data.

    Args:
    df (pandas.DataFrame): Raw data to be preprocessed.

    Returns:
    numpy.ndarray: Preprocessed data.
    """
    # Drop rows with NaN values
    df = df.dropna(subset=['Close'])

    # Select the column to predict
    data_to_use = df['Close'].values

    # Reshape the data
    data_to_use = np.reshape(data_to_use, (-1, 1))

    # Normalize the data
    scaler = MinMaxScaler()
    data_to_use = scaler.fit_transform(data_to_use)

    return data_to_use

# Preprocess data
bitstamp_preprocessed_data = preprocess_data(bitstamp_raw_data)
coinbase_preprocessed_data = preprocess_data(coinbase_raw_data)

# Function to save preprocessed data to a file
def save_preprocessed_data(data, filename):
    """
    Save preprocessed data to a numpy file.

    Args:
    data (numpy.ndarray): Preprocessed data to be saved.
    filename (str): Name of the file to save.
    """
    np.save(filename, data)

# Save preprocessed data
save_preprocessed_data(bitstamp_preprocessed_data, 'bitstamp_preprocessed_data.npy')
save_preprocessed_data(coinbase_preprocessed_data, 'coinbase_preprocessed_data.npy')

# Load preprocessed data
bitstamp_preprocessed_data = np.load('bitstamp_preprocessed_data.npy')
coinbase_preprocessed_data = np.load('coinbase_preprocessed_data.npy')

# Check for NaN values in the preprocessed data
print("Checking for NaN values in preprocessed data...")
print("Bitstamp preprocessed data has NaN values:", np.isnan(bitstamp_preprocessed_data).any())
print("Coinbase preprocessed data has NaN values:", np.isnan(coinbase_preprocessed_data).any())

# Print preprocessed data
print("Bitstamp preprocessed data:")
print(bitstamp_preprocessed_data)
print("Coinbase preprocessed data:")
print(coinbase_preprocessed_data)
