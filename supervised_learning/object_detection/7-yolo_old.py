#!/usr/bin/env python3
"""
Process Outputs
"""
import tensorflow as tf
import numpy as np
import cv2
import os

class Yolo:
    """Define the YOLO class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize attributes and variables"""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [class_name.strip() for class_name in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, array):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-array))

    def load_images(self, folder_path):
        """Load images from the given path"""
        image_paths = []
        images = []

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            if image_path and os.path.isfile(image_path):
                image_paths.append(image_path)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)

        return images, image_paths

    def preprocess_images(self, images):
        """Resize and rescale images"""
        pimages = []
        input_width, input_height = self.model.input.shape[1:3]

        for image in images:
            pimage = cv2.resize(image, (input_width, input_height),
                                interpolation=cv2.INTER_CUBIC)
            pimage = pimage / 255.0
            pimages.append(pimage)

        return np.array(pimages)

    def process_outputs(self, outputs, grid_width, grid_height, anchor_boxes):
        """Process single-image predictions"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height = output.shape[0]
            grid_width = output.shape[1]
            anchor_boxes = output.shape[2]
            # Rest of the method remains the same...

    def filter_boxes(self, box_confidences, box_class_probs, boxes):
        """Filter boxes based on objectness score"""
        box_scores = []
        box_classes = []
        filtered_boxes = []

        for i, (box_confidence, box_class_prob, box) in enumerate(
                zip(box_confidences, box_class_probs, boxes)):
            box_scores_per_output = box_confidence * box_class_prob
            # Rest of the method remains the same...

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-maximum suppression to remove overlapping boxes"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for box_class in np.unique(box_classes):
            indices = np.where(box_classes == box_class)[0]
            filtered_boxes_subset = filtered_boxes[indices]
            box_classes_subset = box_classes[indices]
            box_scores_subset = box_scores[indices]
            # Rest of the method remains the same...

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display image with bounding boxes, class names, and scores"""
        for i, box in enumerate(boxes):
            x_1, y_1, x_2, y_2 = box
            start_point = (int(x_1), int(y_1))
            end_point = (int(x_2), int(y_2))
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(image, start_point, end_point, color, thickness)

            text = f"{self.class_names[box_classes[i]]} {box_scores[i]:.2f}"
            org = (int(x_1), int(y_1) - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 0, 255)
            thickness = 1
            lineType = cv2.LINE_AA
            bottomLeftOrigin = False
            cv2.putText(image, text, org, font, fontScale, color,
                        thickness, lineType, bottomLeftOrigin)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists('detections'):
                   os.makedirs('detections')
            cv2.imwrite('detections/' + file_name, image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Perform object detection on images in folder_path"""
        predictions = []

        images, image_paths = self.load_images(folder_path)
        pimages = self.preprocess_images(images)
        all_outputs = self.model.predict(pimages)

        for i, image in enumerate(images):
            outputs = [all_outputs[x][i, ...] for x in range(len(all_outputs))]
            grid_width = outputs[0].shape[1]
            grid_height = outputs[0].shape[0]
            anchor_boxes = outputs[0].shape[2]
            boxes, box_confidences, box_class_probs = self.process_outputs(
                outputs, grid_width, grid_height, anchor_boxes)
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                box_confidences, box_class_probs, boxes)
            box_predictions, predicted_box_classes, predicted_box_scores = (
                self.non_max_suppression(filtered_boxes, box_classes, box_scores))
            file_name = image_paths[i].split('/')[-1]
            predictions.append((box_predictions,
                                predicted_box_classes, predicted_box_scores))
            self.show_boxes(image, box_predictions,
                            predicted_box_classes, predicted_box_scores,
                            file_name)

        return predictions, image_paths

