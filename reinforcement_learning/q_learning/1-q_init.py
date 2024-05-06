#!/usr/bin/env python3
"""
a function def q_init(env): that initializes the Q-table
"""
import numpy as np
import gym


def q_init(env):
    """
    Initializes the Q-table.

    Args:
        env (gym.Env): The environment instance.

    Returns:
        np.ndarray: The initialized Q-table with dimensions
        (number of states, number of actions).
    """
    # Initialize the Q-table using numpy's zeros function
    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))

    return q_table
