#!/usr/bin/env python3
"""
Play a game using an agent trained by the Deep Q-Learning (DQN) algorithm.
"""

import gym  # Importing the Gym library for reinforcement learning environments
import numpy as np  # Importing NumPy for numerical computations
from tensorflow.keras.models import load_model  # Importing load_model to load the policy network

# Load the trained policy network
policy_network = load_model('/content/policy.h5')  # Loading the policy network saved in policy.h5

# Set up the environment
env = gym.make('CartPole-v1')  # Creating the CartPole-v1 environment

# Play a game using the trained agent
state = env.reset()  # Resetting the environment
done = False  # Initializing done flag
total_reward = 0  # Initializing total reward

while not done:
    env.render()  # Rendering the environment
    state = np.expand_dims(state, axis=0)  # Adding batch dimension
    action = np.argmax(policy_network.predict(state))  # Selecting action with the highest Q-value
    next_state, reward, done, _ = env.step(action)  # Taking action in the environment
    total_reward += reward  # Updating total reward
    state = next_state  # Updating the current state

# Print the total reward obtained
print(f"Total Reward: {total_reward}")

# Close the environment
env.close()  # Closing the environment after the game
