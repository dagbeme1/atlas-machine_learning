import os
import numpy as np

file_path = '../data/Binary_Train.npz'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"No such file or directory: '{file_path}'")

# Load the file if it exists
lib_train = np.load(file_path)
