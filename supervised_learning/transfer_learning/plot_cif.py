#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_cifar10_dataset():
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    return train_images, train_labels

def plot_images(images, labels, num_rows=4, num_cols=8):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    fig.suptitle('cifar10.h5 Dataset')

    for i in range(num_rows):
        for j in range(num_cols):
            idx = np.random.randint(len(images))
            img = images[idx]
            label = labels[idx][0]

            axes[i, j].imshow(img)
            axes[i, j].set_title(f'Label: {label}')
            axes[i, j].axis('off')

    plt.show()

if __name__ == "__main__":
    images, labels = load_cifar10_dataset()
    plot_images(images, labels)
