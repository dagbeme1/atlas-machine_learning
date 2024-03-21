#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import math

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense

preprocess_data = __import__('preprocess_data').preprocess_data


def plot_0(df, title):
    """Function that plots Price at Close vs. Timestamp"""

    plt.figure(figsize=(8,6))
    plt.plot(df)
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel('Price at Close (USD)')
    plt.show()

def plot_1(history, title):
    """Function that plots the loss results of the model"""

    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], 'o-', mfc='none',
             markersize=10, label='Train')
    plt.plot(history.history['val_loss'], 'o-', mfc='none',
             markersize=10, label='Valid')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_2(data_24h, single_label, single_prediction, title):
    """Function that plots a single-step price prediction following
    24h of data"""

    time_steps = list(range(24))
    next_step = 24

    plt.figure(figsize=(8, 6))
    plt.plot(time_steps, data_24h, 'o-', markersize=8, label='data_24h')
    plt.plot(next_step, single_label, 'b+', mfc='none',
             markersize=12, label='Label')
    plt.plot(next_step, single_prediction, 'ro', mfc='none',
             markersize=12, label='Prediction')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Price at Close (Standardized Data)')
    plt.legend()
    plt.show()

def plot_3(future, prediction, title):
    """Function that plots predictions over "batch_size" x 24h
    timeframes"""

    days = list(range(1, future.shape[0] + 1))
    plt.figure(figsize=(12, 6))
    plt.plot(days, future, 'o-', markersize=5, mfc='none', label='Labels')
    plt.plot(days, prediction, 'o-', markersize=5,
             mfc='none', label='Predictions')
    plt.title(title)
    plt.xlim([days[0], days[-1]])
    plt.xlabel('24h Steps')
    plt.ylabel('Price at Close (Standardized Data)')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    file_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    dataset, df, x_train, y_train, x_val, y_val = preprocess_data(file_path)

    print("dataset.shape: {}".format(dataset.shape))
    print("df.shape: {}".format(df.shape))
    print("x_train.shape: {}".format(x_train.shape))
    print("y_train.shape: {}".format(y_train.shape))
    print("x_val.shape: {}".format(x_val.shape))
    print("y_val.shape: {}".format(y_val.shape))
    print('=======================')

    batch_size = 256
    buffer_size = x_train.shape[0]

    # Provide an infinite dataset
    train_iterator = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(buffer_size).batch(batch_size).repeat()

    # Provide an infinite dataset
    val_iterator = tf.data.Dataset.from_tensor_slices(
        (x_val, y_val)).batch(batch_size).repeat()

    n_steps = x_train.shape[-2]
    n_features = x_train.shape[-1]

    # Define the model
    model = Sequential()
    model.add(Bidirectional(
        LSTM(64, activation='relu', input_shape=(n_steps, n_features))))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    epochs = 10
    steps_per_epoch = 800
    validation_steps = 80

    # Train with an infinite dataset
    history = model.fit(train_iterator, epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_iterator,
                        validation_steps=validation_steps)

    print('=======================')
    model.summary()

    # Plot “Price at Close vs. Timestamp” from the dataframe
    plot_0(df['Close'], 'BTC: Price at Close vs. Timestamp')

    # Plot the model loss results
    plot_1(history, 'Training / Validation Losses from History')

    # Make a single-step price prediction following 24h of data
    window_num = 0
    for batch_num, (x, y) in enumerate(val_iterator.take(3)):
        title = f'Prediction for Window {window_num}, Batch {batch_num}'
        plot_2(x[window_num, :, -2].numpy(),
             y[window_num].numpy(),
             model.predict(x)[window_num],
             title)

    # Make predictions over "batch_size" x 24h timeframes
    string = 'Predictions over a {} x 24h Timeframe (Batch {})'
    for batch_num, (x, y) in enumerate(val_iterator.take(3)):
        title = string.format(batch_size, batch_num)
        plot_3(y.numpy(),
             model.predict(x).reshape(-1),
             title)
        batch_num += 1

