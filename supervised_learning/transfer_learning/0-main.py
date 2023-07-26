#!/usr/bin/env python3

import tensorflow.keras as K

preprocess_data = __import__('0-transfer').preprocess_data
# Importing the preprocess_data function from the '0-transfer' module. This function is used for preprocessing the CIFAR-10 data.

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase
# Fixing an issue with saving Keras applications by setting the learning phase of Keras backend.

_, (X, Y) = K.datasets.cifar10.load_data()
# Loading the CIFAR-10 dataset and unpacking it into X (images) and Y (labels).

X_p, Y_p = preprocess_data(X, Y)
# Preprocessing the CIFAR-10 data using the preprocess_data function. This function will preprocess the data and one-hot encode the labels.

model = K.models.load_model('cifar10.h5')
# Loading the pre-trained model stored in 'cifar10.h5' from the current working directory.

model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
# Evaluating the pre-trained model on the preprocessed test data X_p and Y_p. The batch_size is set to 128, and the evaluation results (loss and accuracy) are printed with verbosity level 1.
