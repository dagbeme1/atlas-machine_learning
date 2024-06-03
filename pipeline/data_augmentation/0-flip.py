#!/usr/bin/env python3
"""
This module contains a function to flip an image horizontally using TensorFlow.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Parameters:
    image (tf.Tensor): A 3D tensor containing the image to flip. 
                       The shape of the tensor should be [height, width, channels].

    Returns:
    tf.Tensor: The horizontally flipped image.
    """
    return tf.image.flip_left_right(image)
