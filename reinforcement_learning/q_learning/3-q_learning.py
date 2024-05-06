#!/usr/bin/env python3
"""
the function def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
that performs Q-learning
"""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning.

    Args:
        env: The environment instance.
        Q: The Q-table.
        episodes: The total number of episodes to train over. Default is 5000.
        max_steps: The maximum number of steps per episode. Default is 100.
        alpha: The learning rate. Default is 0.1.
        gamma: The discount rate. Default is 0.99.
        epsilon: The initial threshold for epsilon greedy. Default is 1.
        min_epsilon: The minimum value that epsilon should decay to. Default is 0.1.
        epsilon_decay: The decay rate for updating epsilon between episodes. Default is 0.05.

    Returns:
        np.ndarray, list: Updated Q-table and list containing the rewards per episode.
    """
    # Importing epsilon_greedy function dynamically
    epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

    rewards_per_episode = []

    # Loop through each episode
    for episode in range(episodes):
        # Reset the environment for each episode
        state = env.reset()
        total_reward = 0

        # Loop through each step within the episode
        for unused in range(max_steps):
            # Choose an action based on epsilon-greedy strategy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(Q[state])  # Greedy action

            # Take a step in the environment based on the chosen action
            new_state, reward, done, _ = env.step(action)

            # Update Q-value using Q-learning formula
            Q[state, action] += alpha * \
                (reward + gamma * np.max(Q[new_state]) - Q[state, action])

            state = new_state
            total_reward += reward

            # Check if the episode is done
            if done:
                break

        # Update epsilon for the next episode
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))
        # Append the total reward of the current episode to the list
        rewards_per_episode.append(total_reward)

    # Return the updated Q-table and list of rewards per episode
    return Q, rewards_per_episode
