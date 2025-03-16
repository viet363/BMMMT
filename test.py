import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# Constants
STATE_SIZE = 10
ACTION_SIZE = 4
BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.001
NUM_EPISODES = 1000
EVAL_EPISODES = 10
TAU = 0.01
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Network Intrusion Environment
class NetworkIntrusionEnv(gym.Env):
    def __init__(self):
        super(NetworkIntrusionEnv, self).__init__()
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)
        self.reset()

    def reset(self):
        self.state = np.random.rand(self.state_size)
        return self.state

    def step(self, action):
        reward = 1 if (action == 1 and self.state[0] > 0.8) else -1 if action == 1 else 0
        self.state = np.random.rand(self.state_size)
        done = np.random.rand() > 0.95
        return self.state, reward, done, {}


# Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.5 / np.sqrt(self.weight_mu.size(1)))

    def reset_noise(self):
        self.weight_epsilon.normal_()

    def forward(self, x):
        return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon)


# DQN Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = NoisyLinear(state_size, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Training Function
def train_model(model, target_model, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    best_action = model(next_states).argmax(dim=1, keepdim=True)
    next_q = target_model(next_states).gather(1, best_action).squeeze(1).detach()
    target_q = rewards + GAMMA * next_q * (1 - dones)

    current_q = model(states).gather(1, actions)
    loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Soft Update Function
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# Initialize environment, models, and optimizer
env = NetworkIntrusionEnv()
model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = optim.Adam(model.parameters(), lr=LR)
memory = deque(maxlen=BUFFER_SIZE)

# Training Loop
epsilon = EPSILON_START
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = model(torch.FloatTensor(state).to(device)).argmax().item()

        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        train_model(model, target_model, optimizer, memory)
        state = next_state
        total_reward += reward

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    soft_update(target_model, model, TAU)

    if episode % 100 == 0:

        # Evaluation Function
        def evaluate_model(env, model, num_episodes=EVAL_EPISODES):
            total_rewards = []
            for _ in range(num_episodes):
                state = env.reset()
                total_reward = 0
                done = False
                while not done:
                    with torch.no_grad():
                        action = model(torch.FloatTensor(state).to(device)).argmax().item()
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                total_rewards.append(total_reward)
            avg_reward = np.mean(total_rewards)
            print(f"Average Reward: {avg_reward}")
            return avg_reward


        evaluate_model(env, model)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
