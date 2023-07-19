#!/usr/bin/env python3

"""CustomInception - Neural Network"""

inception_block = __import__('0-inception_block').inception_block
# Importing the 'inception_block' function from the '0-inception_block' module

def create_custom_inception_network():
    """Custom Inception Network"""
    # A docstring explaining that this function
    # creates a Custom Inception Network.

    input_shape_custom = K.Input(shape=(224, 224, 3))
    # Creating an input tensor with shape (224, 224, 3) using Keras Input.

    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', activation='relu')(input_shape_custom)
    # Adding a 2D convolutional layer with 64 filters,
    # a kernel size of (7, 7),
    # a stride of (2, 2), 'same' padding,
    # & ReLU activation function to the input tensor.

    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)
    # Adding a 2D max-pooling layer with pool size (3, 3), stride (2, 2),
    # 'same' padding to the previous convolutional layer.

    X = K.layers.Conv2D(192, (3, 3), activation='relu', padding='same')(X)
    # Adding another 2D convolutional layer with
    # 192 filters, a kernel size of (3, 3),
    # ReLU activation function, and 'same' padding.

    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)
    # Adding another 2D max-pooling layer with
    # pool size (3, 3), stride (2, 2),
    # and 'same' padding to the previous convolutional layer.

    X = inception_block(X, [64, 96, 128, 16, 32, 32])
    # Adding a custom inception block with the provided
    # filter configuration to the previous layer.

    # Several more custom inception blocks are
    # added with different filter configurations.

    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(7, 7),
                                  padding='valid')(X)
    # Adding an average pooling layer with pool size (7, 7) and stride (7, 7),
    # and 'valid' padding to the previous inception blocks.

    X = K.layers.Dropout(0.4)(X)
    # Adding a dropout layer with a dropout rate of 0.4 to the previous layer

    X = K.layers.Dense(1000, activation='softmax')(X)
    # Adding a dense (fully connected) layer with
    # 1000 units and softmax activation
    # to the previous dropout layer.

    custom_model = K.models.Model(inputs=input_shape_custom, outputs=X)
    # Creating a Keras Model with input_shape_custom
    # as the input and X as the output.

    return custom_model
    # Returning the created Keras model from the function.
