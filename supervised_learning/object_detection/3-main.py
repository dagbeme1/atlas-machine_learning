#!/usr/bin/env python3

# Check if the script is being run as the main program
if __name__ == '__main__':
    # Import necessary libraries
    import numpy as np
    Yolo = __import__('3-yolo').Yolo  # Import Yolo class from '3-yolo' module

    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define anchor points used in bounding box adjustments
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])

    # Create an instance of the Yolo class with specified parameters
    yolo = Yolo('data/yolo.h5', 'data/coco_classes.txt', 0.6, 0.5, anchors)

    # Simulated YOLO model outputs at different scales
    output1 = np.random.randn(13, 13, 3, 85)
    output2 = np.random.randn(26, 26, 3, 85)
    output3 = np.random.randn(52, 52, 3, 85)

    # Process the YOLO model outputs and extract boxes, confidences, and class probabilities
    boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))

    # Filter boxes based on confidences and extract classes and scores
    boxes, box_classes, box_scores = yolo.filter_boxes(boxes, box_confidences, box_class_probs)

    # Apply non-maximum suppression to remove redundant boxes
    boxes, box_classes, box_scores = yolo.non_max_suppression(boxes, box_classes, box_scores)

    # Print the final non-maximum suppressed boxes, classes, and scores
    print('Boxes:', boxes)
    print('Box classes:', box_classes)
    print('Box scores:', box_scores)
