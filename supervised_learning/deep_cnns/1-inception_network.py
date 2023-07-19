#!/usr/bin/env python3
# The shebang line at the beginning tells the system to use the Python 3 interpreter
# to execute this script.

"""contains the inception_network function"""
# A multiline string (docstring) describing that this script contains the inception_network function.

import tensorflow.keras as K
# Importing the TensorFlow Keras module with the alias 'K'.

inception_block = __import__('0-inception_block').inception_block
# Importing the 'inception_block' function from the '0-inception_block' module using the custom import method.

def inception_network():
    """
    Builds the inception network
    :return: the Keras model
    """
    # A docstring explaining that this function builds the inception network.

    initializer = K.initializers.he_normal(seed=None)
    # Initializing the weights using the He normal initializer.

    X = K.Input(shape=(224, 224, 3))
    # Creating an input tensor with shape (224, 224, 3) using TensorFlow Keras Input.

    my_layer = K.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(X)
    # Creating a 2D convolutional layer with 64 filters, a kernel size of (7, 7),
    # a stride of (2, 2), 'same' padding, and ReLU activation function, and applying it to the input tensor.

    # Several more convolutional and max-pooling layers are defined with different configurations.

    my_layer = inception_block(my_layer, [64, 96, 128, 16, 32, 32])
    # Applying the custom inception block with the provided filter configuration to the previous layer.

    # Several more custom inception blocks are applied with different filter configurations.

    my_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(my_layer)
    # Applying an average pooling layer with pool size (7, 7) to the previous layer.

    my_layer = K.layers.Dropout(rate=0.4)(my_layer)
    # Applying a dropout layer with a dropout rate of 0.4 to the previous layer.

    my_layer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)
    # Creating a fully connected (dense) layer with 1000 units and softmax activation,
    # and applying it to the previous layer.

    model = K.models.Model(inputs=X, outputs=my_layer)
    # Creating a TensorFlow Keras Model with X as the input and my_layer as the output.

    return model
    # Returning the created TensorFlow Keras model from the function.
