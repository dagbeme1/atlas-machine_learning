#!/usr/bin/env python3
"""
the function def monte_carlo(env, V, policy, episodes=5000, max_steps=100, 
alpha=0.1, gamma=0.99): that performs the Monte Carlo algorithm
"""

import gym  # Import the OpenAI Gym library
import numpy as np  # Import NumPy library for numerical operations

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform the Monte Carlo algorithm to update the value estimate V.
    
    Args:
        env (gym.Env): The OpenAI environment instance.
        V (numpy.ndarray): The value estimate of shape (s,) for each state s.
        policy (function): A function that takes in a state and returns the next action to take.
        episodes (int): The total number of episodes to train over (default is 5000).
        max_steps (int): The maximum number of steps per episode (default is 100).
        alpha (float): The learning rate (default is 0.1).
        gamma (float): The discount rate (default is 0.99).
    
    Returns:
        numpy.ndarray: The updated value estimate V.
    """
    for iteration in range(episodes):  # Iterate over episodes
        # Reset the environment to get initial state
        state = env.reset()
        
        for t in range(max_steps):  # Iterate over steps in an episode
            # Select and perform an action based on the policy
            action = policy(state)
            next_state, episode_reward, done, info = env.step(action)
            
            # Calculate the return G_t for this time step
            G = episode_reward * (gamma ** t)
            
            # Update the value estimate for the current state
            V[state] += alpha * (G - V[state])
            
            # Move to the next state
            state = next_state
            
            # Check if the episode has ended
            if done:
                break
    
    return V  # Return the updated value estimate
