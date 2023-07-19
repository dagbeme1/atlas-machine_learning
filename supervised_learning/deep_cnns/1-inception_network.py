#!/usr/bin/env python3

"""
Builds the inception network as described in Going Deeper with Convolutions
"""

import tensorflow.keras as K
# Importing the TensorFlow Keras module with the alias 'K'.

inception_block = __import__('0-inception_block').inception_block
# Importing the 'inception_block' function from
# the '0-inception_block' module using the custom import method.

def inception_network():
    """
    Builds the inception network as
    described in Going Deeper with Convolutions
    :return: the Keras model
    """
    # A docstring explaining that this function creates an inception network.

    init = K.initializers.he_normal()
    # Initializing the weights using the He normal initializer.

    input_1 = K.Input(shape=(224, 224, 3))
    # Creating an input tensor with shape
    # (224, 224, 3) using TensorFlow Keras Input.

    conv2d_ = K.layers.Conv2D(
        filters=64,
        kernel_initializer=init,
        activation='relu',
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same'
    )
    # Creating a 2D convolutional layer
    # with 64 filters, a kernel size of (7, 7),
    # a stride of (2, 2), 'same' padding, and ReLU activation function.

    conv2d = conv2d_(input_1)
    # Applying the convolutional layer to the input tensor.

    # Several more convolutional and max-pooling layers
    # are defined with different configurations.

    concatenate = inception_block(max_pooling2d_1,
                                  [64, 96, 128, 16, 32, 32])
    # Applying the custom inception block with the provided
    # filter configuration to the previous layer.

    # Several more custom inception blocks are
    # applied with different filter configurations.

    average_pooling2d_ = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(2, 2),
        padding='valid'
    )
    # Creating an average pooling layer
    # with pool size (7, 7) and stride (2, 2).

    average_pooling2d = average_pooling2d_(concatenate_8)
    # Applying the average pooling layer to the previous layer.

    dropout_ = K.layers.Dropout(rate=.4)
    # Creating a dropout layer with a dropout rate of 40%.

    dropout = dropout_(average_pooling2d)
    # Applying the dropout layer to the previous layer.

    dense_ = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )
    # Creating a fully connected (dense) layer
    # with 1000 units and softmax activation.

    dense = dense_(dropout)
    # Applying the dense layer to the previous layer.

    model = K.models.Model(inputs=input_1, outputs=dense)
    # Creating a TensorFlow Keras Model with
    # input_1 as the input and dense as the output.

    return model
    # Returning the created TensorFlow Keras model from the function.
