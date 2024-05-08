#!/usr/bin/env python3
import gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

# Step 1: Environment Setup
env = gym.make('Breakout-v0')

# Step 2: Preprocessing (if necessary)
def preprocess_state(state):
    """Preprocess the state to match the expected input shape."""
    # Assuming 'state' is a NumPy array of shape (84, 84, 3) for color images
    resized_state = cv2.resize(state, (84, 84))
    gray_state = cv2.cvtColor(resized_state, cv2.COLOR_BGR2GRAY)
    normalized_state = gray_state / 255.0
    preprocessed_state = np.expand_dims(normalized_state, axis=-1)
    return preprocessed_state

# Step 3: Neural Network Architecture
def create_model(input_shape, num_actions):
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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Step 5: DQN Algorithm
class DQNAgent:
    def __init__(self, input_shape, num_actions, replay_buffer_capacity=10000, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.00025):
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
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

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
        self.target_model.set_weights(self.model.get_weights())

# Step 6: Training
def train_agent(agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            total_reward += reward
            agent.replay_buffer.add((state, action, reward, next_state, done))
            agent.train()
            state = next_state

        agent.update_target_model()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Initialize agent
input_shape = env.observation_space.shape
num_actions = env.action_space.n
agent = DQNAgent(input_shape, num_actions)

# Train agent
train_agent(agent)

# Save the final policy network
agent.model.save('policy.h5')
