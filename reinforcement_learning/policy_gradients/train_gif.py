#!/usr/bin/env python3
"""
A function that implements a full training.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import IPython.display as display  # Import IPython display for Jupyter

# Assuming policy and policy_gradient are imported correctly from policy_gradient.py

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

def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training process.

    Parameters:
        env (object): The initial environment.
        nb_episodes (int): Number of episodes used for training.
        alpha (float, optional): The learning rate. Defaults to 0.000045.
        gamma (float, optional): The discount factor. Defaults to 0.98.

    Returns:
        list: All values of the score (sum of all rewards during one episode loop).
    """
    # Initialize weights randomly
    weights = np.random.rand(4, 2)

    # List to store scores for each episode
    scores = []

    # Directory to save the extracted frames
    frames_dir = "/content/drive/MyDrive/image slots"
    os.makedirs(frames_dir, exist_ok=True)

    fig, ax = plt.subplots()
    plt.ion()  # Enable interactive mode
    im = ax.imshow(np.zeros((400, 600, 3), dtype=np.uint8))  # Initialize with an empty frame

    for episode in range(nb_episodes):
        # Reset the environment and get the initial state
        state = env.reset()[None, :]

        # Initialize the score for the current episode
        score = 0
        grads = []
        rewards = []

        while True:
            # Choose an action based on the current state and weights
            action, grad = policy_gradient(state, weights)

            # Execute the chosen action and get the next state, reward, and done status
            next_state, reward, done, _ = env.step(action)

            # Store the gradient and reward
            grads.append(grad)
            rewards.append(reward)

            # Update the state
            state = next_state[None, :]

            # Add the reward to the score
            score += reward

            # Save the frame
            frame = env.render(mode='rgb_array')
            frame_path = os.path.join(frames_dir, f"episode_{episode}_frame_{len(rewards)}.png")
            Image.fromarray(frame).save(frame_path)

            # Update the displayed frame
            im.set_array(frame)
            display.display(plt.gcf())
            display.clear_output(wait=True)

            # Break the loop if the episode is done
            if done:
                break

        # Compute the discounted rewards
        for t in range(len(rewards)):
            G = sum(gamma**i * rewards[t + i] for i in range(len(rewards) - t))
            weights += alpha * G * grads[t]

        # Print the current episode number and score
        print(f"Episode: {episode}, Score: {score}", end="\r", flush=False)

        # Append the score to the list of scores
        scores.append(score)

    # Disable interactive mode
    plt.ioff()

    # Display the animation after training
    display_animation(frames_dir)

    return scores

def display_animation(frames_dir):
    """
    Display animation from saved frames.

    Parameters:
        frames_dir (str): Directory containing the saved frames.
    """
    frame_paths = sorted([os.path.join(frames_dir, fname) for fname in os.listdir(frames_dir) if fname.endswith(('.png', '.jpg'))])
    frames = [Image.open(frame_path) for frame_path in frame_paths]

    # Create a figure
    fig = plt.figure()

    # Convert the first frame to an array and display it
    im = plt.imshow(frames[0])

    # Update function for animation
    def updatefig(frame):
        im.set_array(frame)
        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, updatefig, frames=frames, interval=100, blit=True)

    # Display the animation
    plt.axis('off')  # Hide axes
    plt.show()

"""
Main file I used to run the code. Takes a long period to 
run but patience please. You can change the nb_episodes from 10000 
to 1000 to speed up the time
"""
#import gym
#from train import train

#env = gym.make('CartPole-v1')

# Call train with 2 arguments (nb_episodes and alpha)
#scores = train(env, 1000, 0.000045, 0.98)

#env.close()

