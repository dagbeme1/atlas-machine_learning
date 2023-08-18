#!/usr/bin/env python3

"""
Process Outputs
"""
import numpy as np  # Import the NumPy library
import tensorflow as tf  # Import the TensorFlow library


class Yolo:
    """Define the YOLO class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Define and initialize attributes and variables"""
        self.model = tf.keras.models.load_model(
            model_path)  # Load the YOLO model

        # Open the file containing class names
        with open(classes_path, 'r') as f:
            # Read the lines from the file and store class names in a list
            self.class_names = [class_name[:-1] for class_name in f]

        self.class_t = class_t  # Set the threshold for classifying objects
        self.nms_t = nms_t  # Set the threshold for non-maximum suppression
        self.anchors = anchors  # Store the anchor box dimensions

    def process_outputs(self, outputs, image_size):
        """Function that processes single-image predictions"""

        boxes = []  # Store bounding box coordinates
        box_confidences = []  # Store bounding box confidences
        box_class_probs = []  # Store class probabilities

        # Loop over the output feature maps (13x13, 26x26, 52x52)
        for i, output in enumerate(outputs):
            # Get the height of the output feature map
            grid_height = output.shape[0]
            # Get the width of the output feature map
            grid_width = output.shape[1]
            anchor_boxes = output.shape[2]  # Get the number of anchor boxes

            # Extract raw bounding box predictions
            boxs = output[..., :4]

            # Extract network output predictions (x, y, w, h)
            t_x = boxs[..., 0]
            t_y = boxs[..., 1]
            t_w = boxs[..., 2]
            t_h = boxs[..., 3]

            # Calculate grid cell center coordinates
            c_x = np.arange(grid_width).reshape(1, grid_width)
            c_x = np.repeat(c_x, grid_height, axis=0)
            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)

            c_y = np.arange(grid_width).reshape(1, grid_width)
            c_y = np.repeat(c_y, grid_height, axis=0).T
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            # Calculate bounding box coordinates
            b_x = (self.sigmoid(t_x) + c_x) / grid_width
            b_y = (self.sigmoid(t_y) + c_y) / grid_height

            # Calculate bounding box dimensions
            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]
            image_width = self.model.input.shape[1]
            image_height = self.model.input.shape[2]
            b_w = (anchor_width * np.exp(t_w)) / image_width
            b_h = (anchor_height * np.exp(t_h)) / image_height

            # Calculate top-left and bottom-right corner coordinates
            x_1 = b_x - b_w / 2
            y_1 = b_y - b_h / 2
            x_2 = x_1 + b_w
            y_2 = y_1 + b_h

            # Convert coordinates to the original image size
            x_1 *= image_size[1]
            y_1 *= image_size[0]
            x_2 *= image_size[1]
            y_2 *= image_size[0]

            # Update bounding box coordinates
            boxs[..., 0] = x_1
            boxs[..., 1] = y_1
            boxs[..., 2] = x_2
            boxs[..., 3] = y_2

            # Append bounding box coordinates, confidences, and class
            # probabilities
            boxes.append(boxs)
            box_confidence = output[..., 4:5]
            box_confidence = self.sigmoid(box_confidence)
            box_confidences.append(box_confidence)
            classes = output[..., 5:]
            classes = self.sigmoid(classes)
            box_class_probs.append(classes)

        return (boxes, box_confidences, box_class_probs)

    def sigmoid(self, array):
        """Define the sigmoid activation function"""
        return 1 / (1 + np.exp(-1 * array))
