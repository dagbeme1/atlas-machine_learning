#!/usr/bin/env python3
"""
This module contains a function to perform a random crop on an image using TensorFlow.
"""

import tensorflow as tf  # Import the TensorFlow library

def crop_image(image, size):
    """
    Performs a random crop on an image.

    Parameters:
    image (tf.Tensor): A 3D tensor containing the image to crop. 
                       The shape of the tensor should be [height, width, channels].
    size (list or tuple): A list or tuple specifying the size of the output cropped image.
                          It should be in the form [new_height, new_width, channels].

    Returns:
    tf.Tensor: The cropped image with the specified size.
    """
    # Perform a random crop on the image with the given size
    crop_image = tf.image.random_crop(image, size)
    return crop_image  # Return the crop image
