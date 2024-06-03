#!/usr/bin/env python3
"""
shear_image module
This module contains a function to randomly shear an image using TensorFlow.
"""

import tensorflow as tf  # Import the TensorFlow library

def shear_image(image, intensity):
    """
    Randomly shears an image.

    Parameters:
    image (tf.Tensor): A 3D tensor containing the image to shear. 
                       The shape of the tensor should be [height, width, channels].
    intensity (float): The intensity with which the image should be sheared.

    Returns:
    tf.Tensor: The sheared image.
    """
    # Convert the image tensor to a NumPy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Apply random shear to the image array
    sheared_image = tf.keras.preprocessing.image.random_shear(image_array, intensity,
                                                        row_axis=0, col_axis=1,
                                                        channel_axis=2)

    # Convert the sheared image array back to a tensor
    return tf.keras.preprocessing.image.array_to_img(sheared_image)
