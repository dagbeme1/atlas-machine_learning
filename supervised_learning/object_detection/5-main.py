#!/usr/bin/env python3

# Check if the script is being run as the main program
if __name__ == '__main__':
    # Import required libraries
    import cv2  # OpenCV for image processing
    import numpy as np  # NumPy for numerical operations
    Yolo = __import__('5-yolo').Yolo  # Dynamic import of Yolo class

    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define anchor boxes as a NumPy array
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])

    # Create an instance of the Yolo class with specified parameters
    yolo = Yolo('data/yolo.h5', 'data/coco_classes.txt', 0.6, 0.5, anchors)

    # Load images and their paths from a directory using the load_images method
    images, image_paths = yolo.load_images('data/yolo')

    # Preprocess the loaded images using the preprocess_images method
    pimages, image_shapes = yolo.preprocess_images(images)

    # Print the type and shape of the preprocessed images and image shapes
    print(type(pimages), pimages.shape)
    print(type(image_shapes), image_shapes.shape)

    # Generate a random index 'i'
    i = np.random.randint(0, len(images))

    # Print the shape of a randomly selected original image and its corresponding preprocessed image shape
    print(images[i].shape, ':', image_shapes[i])

    # Display the preprocessed image using OpenCV's imshow
    cv2.imshow(image_paths[i], pimages[i])

    # Wait for a key press before closing the display window
    cv2.waitKey(0)

    # Close all OpenCV display windows
    cv2.destroyAllWindows()
