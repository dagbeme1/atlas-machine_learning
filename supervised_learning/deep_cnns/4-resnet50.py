#!/usr/bin/env python3

"""
ResNet-50
"""

# Import the Keras library as K
import tensorflow.keras as K

# Import the identity_block and projection_block functions from the
# respective modules
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block

# Define a function called resnet50 that builds a ResNet-50 network


def resnet50():
    """
    Function that builds a ResNet-50 network as described
    in Deep Residual Learning for Image Recognition (2015)
    """

    # Use the He normal initializer for the weights
    initializer = K.initializers.he_normal()

    # Define the input tensor with shape (224, 224, 3)
    X = K.Input(shape=(224, 224, 3))

    # Layer 1: 7x7 Convolution with 64 filters, stride 2, and no activation
    # function applied yet
    layer_1 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              padding='same',
                              strides=2,
                              kernel_initializer=initializer,
                              activation=None)
    output_1 = layer_1(X)

    # Batch normalization and ReLU activation for Layer 1 output
    norm_1 = K.layers.BatchNormalization()
    output_1 = norm_1(output_1)
    activ_1 = K.layers.Activation('relu')
    output_1 = activ_1(output_1)

    # Layer 2: 3x3 MaxPooling with stride 2
    layer_2 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)
    output_2 = layer_2(output_1)

    # Apply projection block and two identity blocks to output_2
    output_2 = projection_block(output_2, [64, 64, 256], s=1)
    output_2 = identity_block(output_2, [64, 64, 256])
    output_2 = identity_block(output_2, [64, 64, 256])

    # Apply projection block and three identity blocks to output_3
    output_3 = projection_block(output_2, [128, 128, 512], s=2)
    output_3 = identity_block(output_3, [128, 128, 512])
    output_3 = identity_block(output_3, [128, 128, 512])
    output_3 = identity_block(output_3, [128, 128, 512])

    # Apply projection block and five identity blocks to output_4
    output_4 = projection_block(output_3, [256, 256, 1024], s=2)
    output_4 = identity_block(output_4, [256, 256, 1024])
    output_4 = identity_block(output_4, [256, 256, 1024])
    output_4 = identity_block(output_4, [256, 256, 1024])
    output_4 = identity_block(output_4, [256, 256, 1024])
    output_4 = identity_block(output_4, [256, 256, 1024])

    # Apply projection block and two identity blocks to output_5
    output_5 = projection_block(output_4, [512, 512, 2048], s=2)
    output_5 = identity_block(output_5, [512, 512, 2048])
    output_5 = identity_block(output_5, [512, 512, 2048])

    # Apply average pooling with pool_size=7 to output_5
    avg_pool = K.layers.AvgPool2D(pool_size=7,
                                  padding='same',
                                  strides=None)
    output_6 = avg_pool(output_5)

    # Define a dense layer with softmax activation for final classification
    softmax = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2())
    output_7 = softmax(output_6)

    # Instantiate a model from the Model class with input X & output output_7
    model = K.models.Model(inputs=X, outputs=output_7)

    return model
