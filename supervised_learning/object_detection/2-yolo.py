#!/usr/bin/env python3
"""
Yolo Class for Object Detection
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class for performing object detection using YOLO v3 algorithm
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the YOLO object.

        Parameters:
        - model_path (str): Path to the Darknet Keras model.
        - classes_path (str): Path to the file containing class names used by the model.
        - class_t (float): Box score threshold for the initial filtering step.
        - nms_t (float): IOU (Intersection over Union) threshold for non-max suppression.
        - anchors (numpy.ndarray): Array containing anchor box dimensions.

        Attributes:
        - model (tensorflow.keras.Model): Loaded YOLO model.
        - class_names (list): List of class names used by the model.
        - class_t (float): Box score threshold.
        - nms_t (float): Non-max suppression threshold.
        - anchors (numpy.ndarray): Anchor box dimensions.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        Calculate the sigmoid activation function.

        Parameters:
        - x (numpy.ndarray): Input array.

        Returns:
        - numpy.ndarray: Result of the sigmoid activation applied to the input array.
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process single-image predictions from the YOLO model.

        Parameters:
        - outputs (list of numpy.ndarray): Predictions from the Darknet model for a single image.
        - image_size (numpy.ndarray): Original size of the image [image_height, image_width].

        Returns:
        - Tuple of (boxes, box_confidences, box_class_probs):
          - boxes: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 4)
                   containing the processed boundary boxes for each output.
          - box_confidences: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 1)
                             containing the box confidences for each output.
          - box_class_probs: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, classes)
                            containing the box’s class probabilities for each output.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        # Loop over the output feature maps
        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box coordinates and dimensions
            box_xy = self.sigmoid(output[..., :2])
            box_wh = np.exp(output[..., 2:4])
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_probs_raw = output[..., 5:]

            # Create grid
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_x = np.repeat(grid_x, grid_height, axis=0)
            grid_x = np.repeat(grid_x, anchor_boxes, axis=2)

            # Calculate sigmoid transformation for grid coordinates
            b_x = (box_xy[..., 0] + grid_x) / grid_width

            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            grid_y = np.repeat(grid_y, grid_width, axis=1)
            grid_y = np.repeat(grid_y, anchor_boxes, axis=2)

            b_y = (box_xy[..., 1] + grid_y) / grid_height

            # Calculate box width and height
            b_w = box_wh[..., 0] * self.anchors[i, :, 0] / image_size[1]
            b_h = box_wh[..., 1] * self.anchors[i, :, 1] / image_size[0]

            # Calculate box coordinates in the original image
            x_1 = (b_x - b_w / 2) * image_size[1]
            y_1 = (b_y - b_h / 2) * image_size[0]
            x_2 = (b_x + b_w / 2) * image_size[1]
            y_2 = (b_y + b_h / 2) * image_size[0]

            # Append results to the lists
            boxes.append(np.stack([x_1, y_1, x_2, y_2], axis=-1))
            box_confidences.append(box_confidence)
            box_class_probs.append(self.sigmoid(box_class_probs_raw))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter bounding boxes based on box confidence and class probability.

        Parameters:
        - boxes (list of numpy.ndarrays): Processed boundary boxes.
        - box_confidences (list of numpy.ndarrays): Processed box confidences.
        - box_class_probs (list of numpy.ndarrays): Processed box class probabilities.

        Returns:
        - Tuple of (filtered_boxes, box_classes, box_scores):
          - filtered_boxes: A numpy.ndarray of shape (?, 4) containing all of the filtered bounding boxes.
          - box_classes: A numpy.ndarray of shape (?,) containing the class number that each box in filtered_boxes predicts.
          - box_scores: A numpy.ndarray of shape (?) containing the box scores for each box in filtered_boxes.
        """
        obj_thresh = self.class_t

        # Initialize lists to store filtered data
        filtered_boxes = []
        box_classes = []
        box_scores = []

        # Loop over each output
        for i, (box_confidence, box_class_prob, box) in enumerate(
                zip(box_confidences, box_class_probs, boxes)):
            # Compute the box scores for each output feature map
            box_scores_per_output = box_confidence * box_class_prob
            max_box_scores = np.max(box_scores_per_output, axis=3).reshape(-1)

            # Determine the object class of the boxes with the max scores
            max_box_classes = np.argmax(
                box_scores_per_output, axis=3).reshape(-1)

            # Combine all the boxes in a 2D np.ndarray
            box = box.reshape(-1, 4)

            # Create the list of indices pointing to the elements
            # to be removed using the class_t (box score threshold)
            index_list = np.where(max_box_scores < obj_thresh)

            # Delete elements by index
            max_box_scores_filtered = np.delete(max_box_scores, index_list)
            max_box_classes_filtered = np.delete(max_box_classes, index_list)
            filtered_box = np.delete(box, index_list, axis=0)

            # Append the updated arrays to the respective lists
            box_scores.append(max_box_scores_filtered)
            box_classes.append(max_box_classes_filtered)
            filtered_boxes.append(filtered_box)

        # Concatenate the np.ndarrays of all the output feature maps
        box_scores = np.concatenate(box_scores)
        box_classes = np.concatenate(box_classes)
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)

        return filtered_boxes, box_classes, box_scores

