import os
import zipfile
import pandas as pd
import numpy as np
import chardet
from sklearn.preprocessing import MinMaxScaler

def unzip_file(zip_path, extract_path):
    """
    Unzips a file to a specified directory.

    Args:
    zip_path (str): Path to the zip file.
    extract_path (str): Directory where the zip file will be extracted.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df, filename):
    """
    Preprocesses the data by dropping NaN values, selecting the 'Close' column,
    reshaping, and normalizing the data.

    Args:
    df (pandas.DataFrame): Raw data to be preprocessed.
    filename (str): Name of the file to save the preprocessed data.    
    """
    df = df.dropna(subset=['Close'])
    data_to_use = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_to_use = scaler.fit_transform(data_to_use)
    np.save(filename, data_to_use)

def load_preprocessed_data(filename):
    """Loads preprocessed data from a CSV file using pandas and converts it back to a NumPy array."""
    df_loaded = pd.read_csv(filename)
    # Convert the pandas DataFrame back to a NumPy array
    data_loaded = df_loaded.values
    return data_loaded

def main():
    # Paths to zipped files
    bitstamp_zip = '/content/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv.zip'
    coinbase_zip = '/content/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip'

    # Loading data
    bitstamp_raw_data = load_data('/content/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
    coinbase_raw_data = load_data('/content/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

    # Preprocessing data
    preprocess_data(bitstamp_raw_data, 'bitstamp_preprocessed_data.csv.npy')
    preprocess_data(coinbase_raw_data, 'coinbase_preprocessed_data.csv.npy')

    # Loading preprocessed data
    bitstamp_preprocessed_data_v = np.load('bitstamp_preprocessed_data.csv.npy')
    # Convert the NumPy array to a pandas DataFrame
    bitstamp_preprocessed_data_v_df = pd.DataFrame(bitstamp_preprocessed_data_v, columns=['Close'])

    coinbase_preprocessed_data_v = np.load('coinbase_preprocessed_data.csv.npy')
    # Convert the NumPy array to a pandas DataFrame
    coinbase_preprocessed_data_v_df = pd.DataFrame(coinbase_preprocessed_data_v, columns=['Close'])

    # Display the first few rows and shape for Bitstamp data
    print("Bitstamp Data Loaded:")
    print(bitstamp_preprocessed_data_v_df.head())
    print("Shape:", bitstamp_preprocessed_data_v_df.shape)

    # Checking for NaN values in the preprocessed data
    print("Checking for NaN values in preprocessed data...")
    print("Bitstamp preprocessed data has NaN values:", pd.isna(bitstamp_preprocessed_data_v_df).any().any())

    # Repeat the process for Coinbase data
    print("Coinbase Data Loaded:")
    print(coinbase_preprocessed_data_v_df.head())
    print("Shape:", coinbase_preprocessed_data_v_df.shape)
    print("Coinbase preprocessed data has NaN values:", pd.isna(coinbase_preprocessed_data_v_df).any().any())

    # Printing preprocessed data
    print(bitstamp_preprocessed_data_v)
    print(coinbase_preprocessed_data_v)

if __name__ == "__main__":
    main()
