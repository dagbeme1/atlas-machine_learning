#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot  # Import the one_hot function from module '3-one_hot'
load_model = __import__('9-model').load_model  # Import the load_model function from module '9-model'
test_model = __import__('12-test').test_model  # Import the test_model function from module '12-test'


if __name__ == '__main__':
    datasets = np.load('data/MNIST.npz')  # Load the MNIST dataset
    X_test = datasets['X_test']  # Get the test input data from the dataset
    X_test = X_test.reshape(X_test.shape[0], -1)  # Reshape the test input data
    Y_test = datasets['Y_test']  # Get the test labels from the dataset
    Y_test_oh = one_hot(Y_test)  # Convert the test labels to one-hot representation

    network = load_model('network2.h5')  # Load the trained network from the file 'network2.h5'
    print(test_model(network, X_test, Y_test_oh))  # Test the network on the test data and labels
