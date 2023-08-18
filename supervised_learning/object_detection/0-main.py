#!/usr/bin/env python3

if __name__ == '__main__':
    # Import the NumPy library and the Yolo class from the '0-yolo' module
    import numpy as np
    Yolo = __import__('0-yolo').Yolo

    # Set a random seed for reproducibility
    np.random.seed(0)
    
    # Define the anchor box dimensions as a NumPy array
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    
    # Create an instance of the Yolo class
    yolo = Yolo('data/yolo.h5', 'data/coco_classes.txt', 0.6, 0.5, anchors)
    
    # Print a summary of the Yolo model's architecture
    yolo.model.summary()
    
    # Print out the class names recognized by the model
    print('Class names:', yolo.class_names)
    
    # Print the class threshold used for classifying objects
    print('Class threshold:', yolo.class_t)
    
    # Print the Non-Maximum Suppression (NMS) threshold
    print('NMS threshold:', yolo.nms_t)
    
    # Print the anchor box dimensions used for predictions
    print('Anchor boxes:', yolo.anchors)
