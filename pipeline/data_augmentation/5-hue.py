#!/usr/bin/env python3
"""
This module contains a function to change the hue of an image using TensorFlow.
"""

import tensorflow as tf  # Import the TensorFlow library

def change_hue(image, delta):
    """
    Changes the hue of an image.

    Parameters:
    image (tf.Tensor): A 3D tensor containing the image to modify. 
                       The shape of the tensor should be [height, width, channels].
    delta (float): The amount to change the hue. The value should be in the range [-0.5, 0.5].

    Returns:
    tf.Tensor: The image with adjusted hue.
    """
    # Apply hue adjustment to the image
    return tf.image.adjust_hue(image, delta)
