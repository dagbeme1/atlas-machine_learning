#!/usr/bin/env python3
"""
This module contains a function to rotate an image by 90 degrees counter-clockwise using TensorFlow.
"""

import tensorflow as tf  # Import the TensorFlow library

def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Parameters:
    image (tf.Tensor): A 3D tensor containing the image to rotate. 
                       The shape of the tensor should be [height, width, channels].

    Returns:
    tf.Tensor: The rotated image.
    """
    # Rotate the image by 90 degrees counter-clockwise
    return tf.image.rot90(image, k=1)
