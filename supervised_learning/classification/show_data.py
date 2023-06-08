#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import zipfile

zip_path = 'data/Binary_Train.zip'

zip_file = zipfile.ZipFile(zip_path, 'r')

npz_file = zip_file.open('Binary_Train.npz')

lib_train = np.load(npz_file)
X_3D, Y = lib_train['X'], lib_train['Y']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
