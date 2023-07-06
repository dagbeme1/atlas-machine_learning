#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
predict = __import__('13-predict').predict


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']  # Load the test data from the MNIST dataset
    X_test = X_test.reshape(X_test.shape[0], -1)  # Reshape the test data to the appropriate shape
    Y_test = datasets['Y_test']  # Load the test labels

    network = load_model('network2.h5')  # Load the pre-trained network model
    Y_pred = predict(network, X_test)  # Make predictions using the loaded model and the test data
    print(Y_pred)  # Print the predicted values
    print(np.argmax(Y_pred, axis=1))  # Print the indices of the predicted values with the highest probability
    print(Y_test)  # Print the actual test labels
