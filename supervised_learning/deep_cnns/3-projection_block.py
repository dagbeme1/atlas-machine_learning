#!/usr/bin/env python3

# Import the Keras library as K
import tensorflow.keras as K

# Import the projection_block function from the '2-projection_block' module
projection_block = __import__('2-projection_block').projection_block

# Function definition for a projection block, which is a building block of the ResNet architecture
def projection_block(A_prev, filters, s=2):
    # Use He normal initializer for the weights
    initializer = K.initializers.he_normal()

    # Layer 1: 1x1 Convolution with 'filters[0]' filters, stride 's', and no activation (linear)
    F11_layer = K.layers.Conv2D(filters=filters[0],
                                kernel_size=1,
                                padding='same',
                                strides=s,
                                kernel_initializer=initializer,
                                activation=None)
    F11_output = F11_layer(A_prev)
    # Batch normalization for Layer 1
    F11_norm = K.layers.BatchNormalization()
    F11_output = F11_norm(F11_output)
    # Activation function (ReLU) for Layer 1
    F11_activ = K.layers.Activation('relu')
    F11_output = F11_activ(F11_output)

    # Layer 2: 3x3 Convolution with 'filters[1]' filters and no activation (linear)
    F3_layer = K.layers.Conv2D(filters=filters[1],
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=initializer,
                               activation=None)
    F3_output = F3_layer(F11_output)
    # Batch normalization for Layer 2
    F3_norm = K.layers.BatchNormalization()
    F3_output = F3_norm(F3_output)
    # Activation function (ReLU) for Layer 2
    F3_activ = K.layers.Activation('relu')
    F3_output = F3_activ(F3_output)

    # Layer 3: 1x1 Convolution with 'filters[2]' filters and no activation (linear)
    F12_layer = K.layers.Conv2D(filters=filters[2],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation=None)
    F12_output = F12_layer(F3_output)
    # Batch normalization for Layer 3
    F12_norm = K.layers.BatchNormalization()
    F12_output = F12_norm(F12_output)

    # Layer 4: 1x1 Convolution with 'filters[2]' filters, stride 's', and no activation (linear)
    F12_bypass_layer = K.layers.Conv2D(filters=filters[2],
                                       kernel_size=1,
                                       padding='same',
                                       strides=s,
                                       kernel_initializer=initializer,
                                       activation=None)
    F12_bypass = F12_bypass_layer(A_prev)
    # Batch normalization for Layer 4
    bypass_norm = K.layers.BatchNormalization()
    F12_bypass = bypass_norm(F12_bypass)

    # Add input (bypass connection) and output
    output = K.layers.Add()([F12_output, F12_bypass])
    # Activate the combined output
    output = K.layers.Activation('relu')(output)

    return output
