#!/usr/bin/env python3

import tensorflow as tf

# Import the build_model and optimize_model functions from their respective modules
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model

if __name__ == '__main__':
    # Build a neural network model with specified parameters
    model = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    
    # Optimize the model with the specified hyperparameters
    optimize_model(model, 0.01, 0.99, 0.9)
    
    # Print the loss function of the model
    print(model.loss)
    
    # Print the metrics used for evaluation
    print(model.metrics)
    
    # Get the optimizer used for training the model
    opt = model.optimizer
    
    # Print the class of the optimizer
    print(opt.__class__)
    
    with tf.Session() as sess:
        # Initialize the variables of the session
        sess.run(tf.global_variables_initializer())
        
        # Print the learning rate, beta1, and beta2 values of the optimizer
        print(sess.run((opt.lr, opt.beta_1, opt.beta_2)))
