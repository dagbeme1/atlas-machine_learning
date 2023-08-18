#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np  # Import the NumPy library
    Yolo = __import__('1-yolo').Yolo  # Import the Yolo class from module '1-yolo'

    np.random.seed(0)  # Set a random seed for reproducibility
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],  # Define anchor boxes for different scales
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    
    # Create an instance of the Yolo class with specified parameters
    yolo = Yolo('data/yolo.h5', 'data/coco_classes.txt', 0.6, 0.5, anchors)
    
    # Generate random output predictions for different feature maps
    output1 = np.random.randn(13, 13, 3, 85)
    output2 = np.random.randn(26, 26, 3, 85)
    output3 = np.random.randn(52, 52, 3, 85)
    
    # Process the generated output predictions using the Yolo class method
    # 'process_outputs' and pass in the image size as [500, 700]
    boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
    
    # Print the processed results
    print('Boxes:', boxes)  # Print the processed bounding box coordinates
    print('Box confidences:', box_confidences)  # Print the processed bounding box confidences
    print('Box class probabilities:', box_class_probs)  # Print the processed class probabilities
