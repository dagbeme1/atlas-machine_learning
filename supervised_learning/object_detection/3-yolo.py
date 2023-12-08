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
        self.model = tf.keras.models.load_model(model_path)  # Load YOLO model using TensorFlow Keras.
        with open(classes_path, 'r') as f:  # Open and read the file containing class names.
            self.class_names = [line.strip() for line in f]  # Create a list of class names.
        self.class_t = class_t  # Set box score threshold.
        self.nms_t = nms_t  # Set non-max suppression threshold.
        self.anchors = anchors  # Set anchor box dimensions.

    def sigmoid(self, x):
        """
        Calculate the sigmoid activation function.

        Parameters:
        - x (numpy.ndarray): Input array.

        Returns:
        - numpy.ndarray: Result of the sigmoid activation applied to the input array.
        """
        return 1 / (1 + np.exp(-x))  # Apply sigmoid activation function.

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
        boxes = []  # Initialize a list to store processed boundary boxes.
        box_confidences = []  # Initialize a list to store box confidences.
        box_class_probs = []  # Initialize a list to store box class probabilities.

        # Loop over the output feature maps
        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape  # Get output shape.

            # Extract box coordinates and dimensions using sigmoid activation
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
        obj_thresh = self.class_t  # Set box score threshold for filtering.

        # Initialize lists to store filtered data
        filtered_boxes = []  # List to store filtered bounding boxes.
        box_classes = []  # List to store predicted class numbers.
        box_scores = []  # List to store box scores.

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

        # Concatenate the lists into numpy arrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def iou(box1, box2):
        """
        Calculate intersection over union (IOU) for two bounding boxes.

        Parameters:
        - box1 (numpy.ndarray): Coordinates of the first box [x1, y1, x2, y2].
        - box2 (numpy.ndarray): Coordinates of the second box [x1, y1, x2, y2].

        Returns:
        - float: Intersection over union (IOU) between the two boxes.
        """
        xi1 = np.maximum(box1[0], box2[0])  # Calculate the maximum x-coordinate of the intersection.
        yi1 = np.maximum(box1[1], box2[1])  # Calculate the maximum y-coordinate of the intersection.
        xi2 = np.minimum(box1[2], box2[2])  # Calculate the minimum x-coordinate of the intersection.
        yi2 = np.minimum(box1[3], box2[3])  # Calculate the minimum y-coordinate of the intersection.
        inter_area = np.maximum(yi2 - yi1, 0) * np.maximum(xi2 - xi1, 0)  # Calculate the area of intersection.

        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])  # Calculate the area of the first box.
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])  # Calculate the area of the second box.
        union_area = box1_area + box2_area - inter_area  # Calculate the area of union.

        iou = inter_area / union_area if union_area > 0 else 0  # Calculate the IOU.

        return iou  # Return the IOU.



    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Perform non-max suppression to filter overlapping bounding boxes.

        Parameters:
        - filtered_boxes (numpy.ndarray): Filtered bounding boxes.
        - box_classes (numpy.ndarray): Class predictions for each filtered box.
        - box_scores (numpy.ndarray): Box scores for each filtered box.

        Returns:
        - Tuple of (box_predictions, predicted_box_classes, predicted_box_scores):
        - box_predictions: A numpy.ndarray of shape (?, 4) containing all of the predicted bounding boxes ordered by
                         class and box score.
        - predicted_box_classes: A numpy.ndarray of shape (?,) containing the class number for box_predictions
                              ordered by class and box score.
        - predicted_box_scores: A numpy.ndarray of shape (?) containing the box scores for box_predictions
                             ordered by class and box score.
        """
        # Initialize lists to store final predictions
        box_predictions = []  # List to store predicted bounding boxes.
        predicted_box_classes = []  # List to store predicted class numbers.
        predicted_box_scores = []  # List to store predicted box scores.

        # Loop through sorted indices
        for i in range(len(filtered_boxes)):
            # Check if the index is within bounds
            if i >= len(box_scores):
                continue

            # Append the box to predictions
            box_predictions.append(filtered_boxes[i])
            predicted_box_classes.append(box_classes[i])
            predicted_box_scores.append(box_scores[i])

            # Calculate IOU between the current box and the rest
            iou_scores = np.array([self.iou(filtered_boxes[i], box) for j, box in enumerate(filtered_boxes) if j != i])

            # Remove indices where IOU is higher than the threshold
            index_to_remove = np.where(iou_scores > self.nms_t)[0]
            filtered_boxes = np.delete(filtered_boxes, index_to_remove, axis=0)
            box_classes = np.delete(box_classes, index_to_remove)
            box_scores = np.delete(box_scores, index_to_remove)

        # Convert lists to numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

