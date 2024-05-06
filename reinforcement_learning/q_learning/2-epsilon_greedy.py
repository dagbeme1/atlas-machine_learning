#!/usr/bin/env python3
"""
Uses epsilon-greedy to determine the next action
"""
import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Args:
        Q (np.ndarray): numpy.ndarray containing the q-table.
        state (int): The current state.
        epsilon (float): The epsilon value to use for the calculation.

    Returns:
        int: The index of the next action.
    """
    # Generate a random number
    random_value = np.random.uniform(0, 1)

    # Choose a random action with probability epsilon
    if random_value < epsilon:
        action = np.random.randint(0, Q.shape[1])
    # Choose the action with the highest Q-value otherwise
    else:
        action = np.argmax(Q[state])

    return action
