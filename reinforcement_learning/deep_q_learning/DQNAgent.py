#!/usr/bin/env python3
"""
Builds a projection block as described in Deep Residual Learning for Image Recognition.
"""

import gym  # Importing the Gym library for reinforcement learning environments
import numpy as np  # Importing NumPy for numerical computations
# Importing Keras for building neural networks
from tensorflow.keras import Sequential
# Importing layers for the neural network
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam  # Importing the Adam optimizer
import random  # Importing the random module for random sampling
from collections import deque  # Importing deque for experience replay buffer

# Step 1: Environment Setup
env = gym.make('Breakout-v0')  # Creating the Breakout environment

# Step 2: Preprocessing (if necessary)
# Implement preprocessing if needed

# Step 3: Neural Network Architecture


def create_model(input_shape, num_actions):
    """
    Builds the neural network model architecture.

    Args:
        input_shape (tuple): The shape of the input data.
        num_actions (int): The number of possible actions.

    Returns:
        keras.Sequential: The neural network model.
    """
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    return model

# Step 4: Experience Replay


class ExperienceReplay:
    """
    Represents the experience replay buffer.
    """

    def __init__(self, capacity):
        """
        Initializes the experience replay buffer.

        Args:
            capacity (int): The maximum capacity of the buffer.
        """
        self.buffer = deque(
            maxlen=capacity)  # Initializing the experience replay buffer

    def add(self, experience):
        """
        Adds an experience to the buffer.

        Args:
            experience (tuple): The experience tuple to add.
        """
        self.buffer.append(experience)  # Adding experience to the buffer

    def sample(self, batch_size):
        """
        Samples a batch from the buffer.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            list: A list of sampled experiences.
        """
        return random.sample(self.buffer,
                             batch_size)  # Sampling batch from the buffer

# Step 5: DQN Algorithm


class DQNAgent:
    """
    Represents a DQN agent.
    """

    def __init__(
            self,
            input_shape,
            num_actions,
            replay_buffer_capacity=10000,
            batch_size=32,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.00025):
        """
        Initializes the DQN agent.

        Args:
            input_shape (tuple): The shape of the input data.
            num_actions (int): The number of possible actions.
            replay_buffer_capacity (int): The capacity of the experience replay buffer.
            batch_size (int): The size of the training batches.
            gamma (float): The discount factor for future rewards.
            epsilon (float): The exploration rate.
            epsilon_decay (float): The decay rate for epsilon.
            epsilon_min (float): The minimum value for epsilon.
            learning_rate (float): The learning rate for the optimizer.
        """
        self.input_shape = input_shape
        self.num_actions = num_actions
        # Initializing the experience replay buffer
        self.replay_buffer = ExperienceReplay(replay_buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.model = create_model(
            input_shape, num_actions)  # Creating the DQN model
        self.target_model = create_model(
            input_shape, num_actions)  # Creating the target DQN model
        # Initializing target model with main model weights
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = Adam(learning_rate)

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def select_action(self, state):
        """
        Selects an action based on the current state.

        Args:
            state (numpy.ndarray): The current state.

        Returns:
            int: The selected action.
        """
        if np.random.rand() <= self.epsilon:
            # Selecting a random action with epsilon probability
            return np.random.choice(self.num_actions)
        q_values = self.model.predict(state)
        # Selecting the action with the highest Q-value
        return np.argmax(q_values[0])

    def train(self):
        """
        Trains the agent using experience replay.
        """
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        # Sampling a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(
            *batch)  # Unzipping the batch

        states = np.concatenate(states)  # Concatenating states
        next_states = np.concatenate(next_states)  # Concatenating next states

        q_values = self.model.predict(states)  # Predicting Q-values for states
        next_q_values = self.target_model.predict(
            next_states)  # Predicting Q-values for next states

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            # Getting the target Q-values for the current state
            target = q_values[i]
            if done:
                # If episode is done, set target to the immediate reward
                target[action] = reward
            else:
                # Else update target using Bellman equation
                target[action] = reward + self.gamma * \
                    np.amax(next_q_values[i])
            q_values[i] = target  # Setting the updated target Q-values

        self.model.fit(
            states,
            q_values,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0)  # Training the model

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon *
            self.epsilon_decay)  # Decaying epsilon

    def update_target_model(self):
        """
        Updates the target model by copying the weights from the main model.
        """
        self.target_model.set_weights(
            self.model.get_weights())  # Updating the target model weights

# Step 6: Training


def train_agent(agent, num_episodes=1000):
    """
    Trains the agent for a specified number of episodes.

    Args:
        agent (DQNAgent): The DQN agent.
        num_episodes (int): The number of episodes to train for.
    """
    for episode in range(num_episodes):
        state = env.reset()  # Resetting the environment for each episode
        state = np.expand_dims(state, axis=0)  # Adding batch dimension
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)  # Selecting an action
            next_state, reward, done, _ = env.step(
                action)  # Taking action in the environment
            next_state = np.expand_dims(
                next_state, axis=0)  # Adding batch dimension
            total_reward += reward  # Updating total reward
            # Adding experience to replay buffer
            agent.replay_buffer.add((state, action, reward, next_state, done))
            agent.train()  # Training the agent
            state = next_state  # Updating the current state

        agent.update_target_model()  # Updating the target model
        # Printing episode information
        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


# Initialize agent
input_shape = env.observation_space.shape
num_actions = env.action_space.n
agent = DQNAgent(input_shape, num_actions)

# Train agent
train_agent(agent)
