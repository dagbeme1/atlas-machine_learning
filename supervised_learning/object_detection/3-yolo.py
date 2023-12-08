#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

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
        self.model = tf.keras.models.load_model(model_path)
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

    def iou(self, box1, box2):
        """
        Calculate intersection over union (IOU) between two bounding boxes.

        Parameters:
        - box1 (numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2].
        - box2 (numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
        - float: Intersection over union (IOU) between the two bounding boxes.
        """
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])

        inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)
        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Perform non-maximum suppression (NMS) on the filtered boxes.

        Parameters:
        - filtered_boxes (numpy.ndarray): Filtered bounding boxes.
        - box_classes (numpy.ndarray): Class numbers for each box.
        - box_scores (numpy.ndarray): Box scores for each box.

        Returns:
        - Tuple of (box_predictions, predicted_box_classes, predicted_box_scores):
          - box_predictions: numpy.ndarray of shape (?, 4) containing all predicted bounding boxes.
          - predicted_box_classes: numpy.ndarray of shape (?,) containing class numbers for predictions.
          - predicted_box_scores: numpy.ndarray of shape (?) containing box scores for predictions.
        """
        if not filtered_boxes.size or not box_classes.size or not box_scores.size:
            # No valid boxes, return empty arrays
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        # Enable eager execution
        tf.config.run_functions_eagerly(True)

        # Convert NumPy arrays to TensorFlow tensors
        filtered_boxes = tf.convert_to_tensor(filtered_boxes, dtype=tf.float32)
        box_classes = tf.convert_to_tensor(box_classes, dtype=tf.int32)
        box_scores = tf.convert_to_tensor(box_scores, dtype=tf.float32)

        # Perform non-max suppression using TensorFlow's function
        selected_indices = tf.image.non_max_suppression(
            filtered_boxes,
            box_scores,
            max_output_size=self.model.output_shape[1],
            iou_threshold=self.nms_t,
            score_threshold=self.class_t
        )

        # Check if selected_indices is empty
        if selected_indices.numpy().size == 0:
            # No boxes selected, return empty arrays
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        # Use TensorFlow's gather function to select relevant indices
        box_predictions = tf.gather(filtered_boxes, selected_indices)
        predicted_box_classes = tf.gather(box_classes, selected_indices)
        predicted_box_scores = tf.gather(box_scores, selected_indices)

        # Convert the results back to NumPy arrays if needed
        box_predictions = box_predictions.numpy()
        predicted_box_classes = predicted_box_classes.numpy()
        predicted_box_scores = predicted_box_scores.numpy()

        # Disable eager execution
        tf.config.run_functions_eagerly(False)

        return box_predictions, predicted_box_classes, predicted_box_scores

