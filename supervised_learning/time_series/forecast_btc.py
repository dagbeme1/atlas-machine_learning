#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_data(file_path, timesteps):
    """
    Load and preprocess the cryptocurrency data.

    Args:
    file_path (str): Path to the cryptocurrency data file.
    timesteps (int): Number of time steps to consider for each sequence.

    Returns:
    tuple: Tuple containing the input features (X) and target labels (y).
    """
    data = np.load(file_path)
    X, y = [], []
    for i in range(len(data)-timesteps-1):
        X.append(data[i:(i+timesteps), :])
        y.append(data[i+timesteps, 0]) # Predicting only the close price
    return np.array(X), np.array(y)

# Load and preprocess Bitstamp data
bitstamp_X, bitstamp_y = load_data('/content/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22', timesteps=60)

# Load and preprocess Coinbase data
coinbase_X, coinbase_y = load_data('/content/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09', timesteps=60)

# Split data into training and validation sets
bitstamp_X_train, bitstamp_X_val, bitstamp_y_train, bitstamp_y_val = train_test_split(bitstamp_X, bitstamp_y, test_size=0.2)
coinbase_X_train, coinbase_X_val, coinbase_y_train, coinbase_y_val = train_test_split(coinbase_X, coinbase_y, test_size=0.2)

# Define LSTM model
def create_model(input_shape):
    """
    Create and compile an LSTM model.

    Args:
    input_shape (tuple): Shape of the input data.

    Returns:
    tensorflow.keras.Model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(20, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))
    adam_optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
    model.compile(optimizer=adam_optimizer, loss='mse')
    return model

# Create Bitstamp model
bitstamp_model = create_model(input_shape=(60, bitstamp_X.shape[2]))

# Train Bitstamp model
bitstamp_history = bitstamp_model.fit(bitstamp_X_train, bitstamp_y_train, epochs=5, validation_data=(bitstamp_X_val, bitstamp_y_val), verbose=1)

# Create Coinbase model
coinbase_model = create_model(input_shape=(60, coinbase_X.shape[2]))

# Train Coinbase model
coinbase_history = coinbase_model.fit(coinbase_X_train, coinbase_y_train, epochs=5, validation_data=(coinbase_X_val, coinbase_y_val), verbose=1)

# Plot training and validation loss
plt.plot(bitstamp_history.history['loss'], label='Bitstamp Train')
plt.plot(bitstamp_history.history['val_loss'], label='Bitstamp Validation')
plt.plot(coinbase_history.history['loss'], label='Coinbase Train')
plt.plot(coinbase_history.history['val_loss'], label='Coinbase Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save models
bitstamp_model.save('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.h5')
coinbase_model.save('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.h5')

print("Models saved")

