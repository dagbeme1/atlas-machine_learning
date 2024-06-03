#!/usr/bin/env python3
"""
a function def pca_color(image, alphas): that performs PCA color augmentation as described in the AlexNet paper
"""
import tensorflow as tf
import numpy as np

def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.

    Parameters:
        image (tf.Tensor): A 3D tf.Tensor containing the image to change.
        alphas (tuple): A tuple of length 3 containing the amount that each channel should change.

    Returns:
        tf.Tensor: The augmented image.
    """
    # Convert the image to a NumPy array
    img_np = tf.squeeze(tf.cast(image, tf.float32)).numpy()

    # Flatten the image and subtract the mean
    flattened = img_np.reshape(-1, 3)
    mean = np.mean(flattened, axis=0)
    centered = flattened - mean

    # Compute the covariance matrix
    cov = np.cov(centered, rowvar=False)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort the eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:,idx]

    # Project the image onto the eigenvectors
    projection = np.dot(centered, eigenvectors.T)

    # Scale the projection by the given alphas
    scaled_projection = projection * alphas

    # Reconstruct the image
    reconstructed = np.dot(scaled_projection, eigenvectors) + mean

    # Reshape the image back to its original shape
    reconstructed_img = reconstructed.reshape(img_np.shape)

    # Convert the image back to a TensorFlow tensor
    return tf.convert_to_tensor(reconstructed_img, dtype=image.dtype)
