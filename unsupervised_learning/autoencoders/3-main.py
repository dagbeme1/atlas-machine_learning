#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('3-variational').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.random.set_seed(0)
encoder, decoder, auto = autoencoder(784, [512], 2)
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded, mu, log_sig = encoder.predict(x_test[:10])
print(mu)
print(np.exp(log_sig / 2))
reconstructed = decoder.predict(encoded).reshape((-1, 28, 28))
x_test = x_test.reshape((-1, 28, 28))

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()

