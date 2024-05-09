#!/usr/bin/env python4
import numpy as np

def play(env, Q, max_steps=100):
    """
    Play a game using the Q-table learned from Q-learning.

    Args:
        env: The environment instance.
        Q (numpy.ndarray): The Q-table.
        max_steps (int): Maximum number of steps per episode.

    Returns:
        int: Total reward obtained during the game.
    """
    total_reward = 0  # Initialize total reward
    state = env.reset()  # Reset the environment and get the initial state
    for step in range(max_steps):
        # Display the current state (assuming env.render() is not suitable)
        print("Current state:", state)

        # Select action based on Q-table (exploit)
        action = np.argmax(Q[state])

        # Execute action and get feedback
        next_state, reward, done, _ = env.step(action)
        total_reward += reward  # Accumulate reward

        # Move to the next state
        state = next_state

        # Break the loop if the goal is reached or max steps are exceeded
        if done:
            break

    return total_reward
