#!/usr/bin/env python3

# Check if the script is being run as the main program
if __name__ == '__main__':
    # Import necessary libraries
    import cv2
    import numpy as np
    Yolo = __import__('4-yolo').Yolo  # Import Yolo class from '4-yolo' module

    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define anchor points used in bounding box adjustments
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])

    # Create an instance of the Yolo class with specified parameters
    yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)

    # Load images and their paths from the specified directory
    images, image_paths = yolo.load_images('../data/yolo')

    # Generate a random index to select a random image from the loaded images
    i = np.random.randint(0, len(images))

    # Display the selected image using OpenCV
    cv2.imshow(image_paths[i], images[i])
    
    # Wait for a key press and close the displayed window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
