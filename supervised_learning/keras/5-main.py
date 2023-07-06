#!/usr/bin/env python3

"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

# Set seed for reproducibility in TensorFlow and NumPy
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)

# Set session configuration for TensorFlow
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('5-train').train_model

if __name__ == '__main__':
    # Load the MNIST dataset
    datasets = np.load('../data/MNIST.npz')
    
    # Load and reshape the training data
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    
    # Perform one-hot encoding on the training labels
    Y_train_oh = one_hot(Y_train)
    
    # Load and reshape the validation data
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    
    # Perform one-hot encoding on the validation labels
    Y_valid_oh = one_hot(Y_valid)

    # Set hyperparameters
    lambtha = 0.0001
    keep_prob = 0.95
    
    # Build the model architecture
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    
    # Set optimizer hyperparameters and optimize the model
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    
    # Set training hyperparameters and train the model
    batch_size = 64
    epochs = 5
    train_model(network, X_train, Y_train_oh, batch_size, epochs, validation_data=(X_valid, Y_valid_oh))
