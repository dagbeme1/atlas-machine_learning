#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

# Load preprocessed data
bitstamp_data = pd.read_csv('/content/bitstamp_preprocessed_data.csv')
coinbase_data = pd.read_csv('/content/coinbase_preprocessed_data.csv')

# Prepare data for LSTM
def prepare_data(data, timesteps):
    X, Y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        Y.append(data.iloc[i + timesteps, 0])
    return np.array(X), np.array(Y)

timesteps = 60
bitstamp_X, bitstamp_y = prepare_data(bitstamp_data, timesteps)
coinbase_X, coinbase_y = prepare_data(coinbase_data, timesteps)

# Split data into training and validation sets
bitstamp_X_train, bitstamp_X_val, bitstamp_y_train, bitstamp_y_val = train_test_split(bitstamp_X, bitstamp_y, test_size=0.2, random_state=42)
coinbase_X_train, coinbase_X_val, coinbase_y_train, coinbase_y_val = train_test_split(coinbase_X, coinbase_y, test_size=0.2, random_state=42)

# Define LSTM model
def create_model(timesteps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(timesteps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Create and train models
bitstamp_model = create_model(timesteps, bitstamp_data.shape[1])
coinbase_model = create_model(timesteps, coinbase_data.shape[1])

bitstamp_history = bitstamp_model.fit(bitstamp_X_train, bitstamp_y_train, epochs=5, batch_size=256, validation_data=(bitstamp_X_val, bitstamp_y_val), verbose=1)
coinbase_history = coinbase_model.fit(coinbase_X_train, coinbase_y_train, epochs=5, batch_size=256, validation_data=(coinbase_X_val, coinbase_y_val), verbose=1)

# Plotting the training & validation loss values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(bitstamp_history.history['loss'])
plt.plot(bitstamp_history.history['val_loss'])
plt.title('Bitstamp Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(coinbase_history.history['loss'], label='train')
plt.plot(coinbase_history.history['val_loss'], label='validation')
plt.title('Coinbase Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Save the models
bitstamp_model.save('bitstamp_model_v2.h5')
coinbase_model.save('coinbase_model_v2.h5')

print("Models saved")
