#!/usr/bin/env python3
"""
This module contains a function to randomly change the brightness of an image using TensorFlow.
"""

import tensorflow as tf  # Import the TensorFlow library

def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Parameters:
    image (tf.Tensor): A 3D tensor containing the image to modify. 
                       The shape of the tensor should be [height, width, channels].
    max_delta (float): The maximum amount to change the brightness. 
                       The value should be in the range [0, 1).

    Returns:
    tf.Tensor: The image with adjusted brightness.
    """
    # Apply random brightness adjustment to the image
    return tf.image.random_brightness(image, max_delta)
