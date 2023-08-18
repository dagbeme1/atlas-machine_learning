#!/usr/bin/env python3
"""
Initialize Yolo
"""
import tensorflow as tf  # Import the TensorFlow library


class Yolo:
    """define the YOLO class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """define and initialize attributes and variables"""
        self.model = tf.keras.models.load_model(
            model_path)  # Load the pre-trained YOLO model

        # Open the file containing class names
        with open(classes_path, 'r') as f:
            # Read the lines from the file and store class names in a list
            self.class_names = [class_name[:-1] for class_name in f]

        self.class_t = class_t  # Set the threshold for classifying objects
        self.nms_t = nms_t  # Set the threshold for non-maximum suppression
        self.anchors = anchors  # Store the anchor box dimensions used for predictions
