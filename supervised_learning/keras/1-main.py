#!/usr/bin/env python3

# Import the build_model function from the '1-input' module
build_model = __import__('1-input').build_model

if __name__ == '__main__':
    # Build a neural network model with specified parameters
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    
    # Print the summary of the model
    network.summary()
    
    # Print the regularization losses of the model
    print(network.losses)