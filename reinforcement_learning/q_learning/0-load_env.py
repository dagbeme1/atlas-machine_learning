#!/usr/bin/env python3
"""
Loads the pre-made FrozenLakeEnv environment from OpenAI’s gym

This script provides a function to load the FrozenLake environment
from OpenAI's Gym. The FrozenLake environment represents
a grid-world game where the objective is for the agent
to navigate from the starting position to the goal position
while avoiding falling into holes.
"""

# Importing required libraries
import numpy as np
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Load the FrozenLake environment.

    Parameters:
        desc (str or None): String representation of the map
        to use. If None, the default map will be used. (Default: None)
        map_name (str or None): Name of the map to load.
        If None, it is ignored. (Default: None)
        is_slippery (bool): Whether to enable slippery mode
        (stochastic transitions). (Default: False)

    Returns:
        env (gym.Env): The FrozenLake environment instance.
    """
    # Create the FrozenLake environment with specified parameters
    env = gym.make(id='FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)

    # returns env
    return env
