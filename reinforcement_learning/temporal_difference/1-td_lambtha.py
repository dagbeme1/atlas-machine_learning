#!/usr/bin/env python3
"""
the function def td_lambtha(env, V, policy, lambtha, episodes=5000, 
max_steps=100, alpha=0.1, gamma=0.99): that performs the TD(λ) algorithm
"""

import gym
import numpy as np

def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform the TD(λ) algorithm.

    Args:
        env: OpenAI environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: Function that takes in a state and returns the next action to take.
        lambtha: Eligibility trace factor.
        episodes: Total number of episodes to train over (default is 5000).
        max_steps: Maximum number of steps per episode (default is 100).
        alpha: Learning rate (default is 0.1).
        gamma: Discount rate (default is 0.99).

    Returns:
        V: The updated value estimate.
    """
    # Initialize eligibility traces
    z = np.zeros_like(V)
    
    for iteration in range(episodes):
        # Reset the environment to get initial state
        state = env.reset()
        
        # Iterate over a maximum number of steps per episode
        for t in range(max_steps):
            # Select and perform an action
            action = policy(state)
            next_state, reward, done, iteration = env.step(action)
            
            # Calculate the target value
            target = reward + gamma * np.max(V[next_state])
            
            # Update the value estimate and eligibility traces
            delta = target - V[state]
            V[state] += alpha * delta
            z *= lambtha * gamma
            z[state] += 1
            
            # Decay eligibility traces
            z *= lambtha
            
            # Move to the next state
            state = next_state
            
            # Check if the episode has ended
            if done:
                break
    
    return V
