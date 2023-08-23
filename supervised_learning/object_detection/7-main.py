#!/usr/bin/env python3

# Import necessary libraries and modules
import numpy as np

# Import the Yolo class from the '7-yolo' module (assuming '7-yolo' is a file name)
Yolo = __import__('7-yolo').Yolo

# Set a random seed for reproducibility
np.random.seed(0)

# Define anchor boxes for YOLO model
anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]])

# Create an instance of the Yolo class with specified parameters
yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)

# Use the Yolo class to predict objects in images located in the specified folder
predictions, image_paths = yolo.predict('data/yolo')

# Iterate through the image paths to find the index of the 'dog.jpg' image
for i, name in enumerate(image_paths):
    if "dog.jpg" in name:
        ind = i
        break

# Print the path of the 'dog.jpg' image
print(image_paths[ind])

# Print the predictions for objects detected in the 'dog.jpg' image
print(predictions[ind])
