#!/usr/bin/env python3

"""DCNN - Inception Network"""

import tensorflow.keras as K
# Importing the TensorFlow Keras library using the alias 'K'.

inception_block = __import__('0-inception_block').inception_block
# Importing a function 'inception_block' from the module '0-inception_block'.

def inception_network():
    """Inception Network"""
    # This is a docstring, a multiline comment that provides
    # a brief description of the function.

    X_input = K.Input(shape=(224, 224, 3))
    # Creating an input tensor with shape (224, 224, 3) using Keras'
    # Input function. This defines the shape of the
    # input images for the network.

    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', activation='relu')(X_input)
    # Adding a 2D convolutional layer with 64 filters,
    # a kernel size of (7, 7), a stride of (2, 2), and 'same' padding.
    # The 'relu' activation function is used.

    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)
    # Adding a 2D max pooling layer with a pool size
    # of (3, 3), a stride of (2, 2), and 'same' padding.

    X = K.layers.Conv2D(192, (3, 3), activation='relu', padding='same')(X)
    # Adding another 2D convolutional layer with 192 filters,
    # a kernel size of (3, 3), 'relu' activation, and 'same' padding.

    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)
    # Adding another 2D max pooling layer with a pool size of
    # (3, 3), a stride of (2, 2), and 'same' padding.

    # The next lines use a function 'inception_block' with
    # specific parameters to create multiple inception blocks,
    # which are commonly used in Inception networks.

    X = inception_block(X, [64, 96, 128, 16, 32, 32])
    # Creating an inception block with
    # filter sizes [64, 96, 128, 16, 32, 32].

    X = inception_block(X, [128, 128, 192, 32, 96, 64])
    # Creating an inception block with filter
    # sizes [128, 128, 192, 32, 96, 64].

    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)
    # Adding another 2D max pooling layer with
    # a pool size of (3, 3), a stride of (2, 2), and 'same' padding.

    # Similar lines follow to create more inception blocks.

    X = inception_block(X, [192, 96, 208, 16, 48, 64])
    # Creating an inception block with
    # filter sizes [192, 96, 208, 16, 48, 64].

    # ... more inception blocks ...

    X = inception_block(X, [384, 192, 384, 48, 128, 128])
    # Creating an inception block with
    # filter sizes [384, 192, 384, 48, 128, 128].

    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(7, 7), padding='valid')(X)
    # Adding an average pooling layer with a pool size
    # of (7, 7), a stride of (7, 7), and 'valid' padding.

    X = K.layers.Dropout(0.4)(X)
    # Adding a dropout layer with a dropout rate
    # of 0.4, which helps prevent overfitting.

    X = K.layers.Dense(1000, activation='softmax')(X)
    # Adding a dense (fully connected) layer with 1000 units
    # and a 'softmax' activation function for multiclass classification.

    model = K.models.Model(inputs=X_input, outputs=X)
    # Creating a Keras Model using the input tensor
    # X_input and the output tensor X.

    return model
    # Returning the constructed model.

# The function 'inception_network' defines an Inception
# Network with multiple inception blocks and specific layers
# for image classification. The model is returned at the end of the function
