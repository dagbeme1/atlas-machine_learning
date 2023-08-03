#!/usr/bin/env python3

import os
import glob

def find_cifar10_h5_file():
    search_pattern = "cifar10.h5"

    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file == search_pattern:
                return os.path.join(root, file)

    return None

if __name__ == "__main__":
    h5_file_path = find_cifar10_h5_file()
    if h5_file_path is not None:
        print(f"Found cifar10.h5 file: {h5_file_path}")
        # Here you can call the necessary functions to graph the cifar10.h5 file
    else:
        print("cifar10.h5 file not found in the current directory or its subdirectories.")
