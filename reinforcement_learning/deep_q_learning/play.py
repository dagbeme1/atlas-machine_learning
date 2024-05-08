#!/usr/bin/env python3
"""
Play a game using an agent trained by the Deep Q-Learning (DQN) algorithm.
"""

import gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

# Step 1: Environment Setup
env = gym.make('CartPole-v1')
# Reset the environment to get initial state
env.reset()

# Now you can render/open the environment
env.render()

# Step 2: Preprocessing (if necessary)
# Implement preprocessing if needed

# Step 3: Neural Network Architecture
def create_model(input_shape, num_actions):
    """
    Creates the neural network model architecture.
    
    Args:
        input_shape (tuple): The shape of the input data.
        num_actions (int): The number of possible actions.
        
    Returns:
        keras.Sequential: The neural network model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
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
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        """
        Adds an experience to the buffer.
        
        Args:
            experience (tuple): The experience tuple to add.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Samples a batch from the buffer.
        
        Args:
            batch_size (int): The size of the batch to sample.
            
        Returns:
            list: A list of sampled experiences.
        """
        return random.sample(self.buffer, batch_size)

# Step 5: DQN Algorithm
class DQNAgent:
    """
    Represents a DQN agent.
    """
    def __init__(self, input_shape, num_actions, replay_buffer_capacity=10000, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.00025):
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
        self.replay_buffer = ExperienceReplay(replay_buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.model = create_model(input_shape, num_actions)
        self.target_model = create_model(input_shape, num_actions)
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
            return np.random.choice(self.num_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self):
        """
        Trains the agent using experience replay.
        """
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = list(states)
        states = np.concatenate(states)

        q_values = self.model.predict(states)
        #q_values = self.model.predict(states)
        next_q_values = []
        for state in next_states:
            next_q_values.append(self.target_model.predict(state))
        #next_q_values = self.target_model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = q_values[i]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(next_q_values[i])
            q_values[i] = target

        self.model.fit(states, q_values, batch_size=self.batch_size, epochs=1, verbose=0)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        """
        Updates the target model weights with the main model weights.
        """
        self.target_model.set_weights(self.model.get_weights())

    def play_game(self):
        """
        Plays a game using the trained agent and displays it in the environment.
        """
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            state = np.expand_dims(state, axis=0)
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            total_reward += reward
            state = next_state

        print(f"Total Reward: {total_reward}")

# Step 6: Training
def train_agent(agent, num_episodes=1000):
    """
    Trains the DQN agent.
    
    Args:
        agent (DQNAgent): The DQN agent to train.
        num_episodes (int): The number of episodes to train for.
    """
    for episode in range(num_episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        done = False
        total_reward = 0

        while not done:
            # Select action and update environment
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Expand dimensions of next state
            next_state = np.expand_dims(next_state, axis=0)

            # Update total reward and replay buffer
            total_reward += reward
            agent.replay_buffer.add((state, action, reward, next_state, done))

            # Train the agent on the batch
            agent.train()

            # Update state
            state = next_state

        agent.update_target_model()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Initialize agent
input_shape = (4,)
num_actions = env.action_space.n
agent = DQNAgent(input_shape, num_actions)

# Train agent
train_agent(agent)

# Save the final policy network
agent.model.save('/content/policy.h5')

# Play a game using the trained agent
agent.play_game()

# Close the environment
env.close()
