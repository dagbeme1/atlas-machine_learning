#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Training history
epochs = list(range(1, 31))
loss_values = [1.9980, 1.4057, 1.2615, 1.1645, 1.1158, 1.0678, 1.0208, 0.9818, 0.9390, 0.9150, 0.8892, 0.8687, 0.8417, 0.8175, 0.8145, 0.7884, 0.7648, 0.7602, 0.7482, 0.7337, 0.7240, 0.7068, 0.6981, 0.6838, 0.6741, 0.6630, 0.6596, 0.6389, 0.6312, 0.6292]
accuracy_values = [0.3628, 0.5207, 0.5681, 0.5946, 0.6142, 0.6268, 0.6400, 0.6528, 0.6726, 0.6808, 0.6880, 0.6920, 0.7022, 0.7073, 0.7140, 0.7202, 0.7301, 0.7319, 0.7331, 0.7370, 0.7426, 0.7457, 0.7531, 0.7629, 0.7630, 0.7693, 0.7638, 0.7756, 0.7714, 0.7761]

# Plotting the loss
plt.plot(epochs, loss_values, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the accuracy
plt.plot(epochs, accuracy_values, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)
plt.show()
