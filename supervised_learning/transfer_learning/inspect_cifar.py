#!/usr/bin/env python3
import h5py

def inspect_h5py_file(h5_file):
    with h5py.File(h5_file, 'r') as file:
        print("Datasets in the file:")
        for name in file:
            print(name)

        print("\nDataset shapes:")
        for name in file:
            dataset = file.get(name)
            if isinstance(dataset, h5py.Dataset):
                print(f"Dataset name: {name}")
                print(f"Shape: {dataset.shape}")
                print("-" * 30)

if __name__ == "__main__":
    h5_file_path = "C:/Users/denni/holbertonschool-machine_learning/supervised_learning/transfer_learning/cifar10.h5"
    inspect_h5py_file(h5_file_path)

