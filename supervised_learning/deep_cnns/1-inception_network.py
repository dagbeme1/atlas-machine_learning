#!/usr/bin/env python3
# The shebang line at the beginning tells the system to use the Python 3 interpreter
# to execute this script.

"""DCNN - Inception Network"""
# A multiline string (docstring) describing the purpose of the script.

import tensorflow.keras as K
# Importing the TensorFlow Keras module with the alias 'K'.

inception_block = __import__('0-inception_block').inception_block
# Importing the 'inception_block' function from the '0-inception_block' module using the custom import method.

def inception_network():
    """Inception Network"""
    # A docstring explaining that this function builds the Inception Network.

    X_input = K.Input(shape=(224, 224, 3))
    # Creating an input tensor with shape (224, 224, 3) using TensorFlow Keras Input.

    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', activation='relu')(X_input)
    # Creating a 2D convolutional layer with 64 filters, a kernel size of (7, 7),
    # a stride of (2, 2), 'same' padding, ReLU activation function, and applying it to the input tensor.

    # Several more convolutional and max-pooling layers are defined with different configurations.

    X = inception_block(X, [64, 96, 128, 16, 32, 32])
    # Applying the custom inception block with the provided filter configuration to the previous layer.

    # Several more custom inception blocks are applied with different filter configurations.

    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(7, 7),
                                  padding='valid')(X)
    # Applying an average pooling layer with pool size (7, 7) and stride (7, 7) to the previous layer.

    X = K.layers.Dropout(0.4)(X)
    # Applying a dropout layer with a dropout rate of 0.4 to the previous layer.

    X = K.layers.Dense(1000, activation='softmax')(X)
    # Creating a fully connected (dense) layer with 1000 units and softmax activation,
    # and applying it to the previous layer.

    model = K.models.Model(inputs=X_input, outputs=X)
    # Creating a TensorFlow Keras Model with X_input as the input and X as the output.

    return model
    # Returning the created TensorFlow Keras model from the function.
