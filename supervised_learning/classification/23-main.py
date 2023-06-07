#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import zipfile

Deep = __import__('23-deep_neural_network').DeepNeuralNetwork

zip_path = 'data/Binary_Train.zip'

zip_file = zipfile.ZipFile(zip_path, 'r')

npz_file = zip_file.open('Binary_Train.npz')

lib_train = np.load(npz_file)
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [5, 3, 1])
A, cost = deep.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = deep.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
