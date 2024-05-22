#!/usr/bin/env python3
"""
a function that computes to policy with a weight of a matrix
"""

import numpy as np  # Importing the numpy library

def policy(matrix, weights):
    """
    Function to compute the policy with a given weight of a matrix.

    Parameters:
        matrix (numpy.ndarray): The input matrix.
        weights (numpy.ndarray): The weights applied to the matrix.

    Returns:
        numpy.ndarray: The computed policy.
    """
    combined = np.dot(matrix, weights)  # Compute the dot product of the matrix and weights
    # Compute softmax
    exp_values = np.exp(combined - np.max(combined))  # Subtract max value for numerical stability and apply exponential function
    softmax_result = exp_values / np.sum(exp_values)  # Normalize to get the softmax probabilities
    return softmax_result  # Return the computed policy


def policy_gradient(state, weights):
    """
    Function to compute the Monte Carlo policy gradient.

    Parameters:
        state (numpy.ndarray): The current state.
        weights (numpy.ndarray): The weights associated with the state.

    Returns:
        tuple: A tuple containing the selected action and the gradient.
    """
    state_matrix = policy(state, weights)  # Compute the policy for the given state and weights
    selected_action = np.random.choice(range(len(state_matrix[0])), p=state_matrix[0])  # Select an action based on the policy probabilities
    # Compute gradient softmax
    reshaped_softmax = state_matrix.reshape(-1, 1)  # Reshape the state matrix for gradient calculation
    gradient_softmax = np.diagflat(reshaped_softmax) - reshaped_softmax @ reshaped_softmax.T  # Compute the softmax gradient
    derivative_state_matrix = gradient_softmax[selected_action, :]  # Extract the gradient for the selected action
    derivative_log = derivative_state_matrix / state_matrix[0, selected_action]  # Compute the derivative of the log-probability
    calculated_gradient = state.T @ derivative_log[None, :]  # Compute the gradient with respect to the state
    return selected_action, calculated_gradient  # Return the selected action and the calculated gradient
