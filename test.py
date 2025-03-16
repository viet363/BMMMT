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

# Mô hình môi trường phát hiện xâm nhập mạng
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
        reward = 0
        done = False
        if action == 1 and self.state[0] > 0.8:  # Nếu tấn công mạnh
            reward = 1
        elif action == 1 and self.state[0] <= 0.8:
            reward = -1  # Chặn nhầm
        self.state = np.random.rand(self.state_size)
        done = np.random.rand() > 0.95  # Xác suất kết thúc episode
        return self.state, reward, done, {}

# Noisy Linear Layer cho exploration hiệu quả
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()

    def forward(self, x):
        return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon)

# Mô hình DQN với Noisy Layers
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

# Khởi tạo môi trường và mô hình
env = NetworkIntrusionEnv()
state_size = env.state_size
action_size = env.action_size
model = DQN(state_size, action_size)
target_model = DQN(state_size, action_size)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = optim.Adam(model.parameters(), lr=LR)
memory = deque(maxlen=BUFFER_SIZE)

# Huấn luyện DQN với Double DQN & PER
def train_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    best_action = model(next_states).argmax(dim=1, keepdim=True)
    next_q = target_model(next_states).gather(1, best_action).squeeze(1).detach()
    target_q = rewards + GAMMA * next_q * (1 - dones)

    current_q = model(states).gather(1, actions)
    loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Huấn luyện mô hình
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state)
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        train_model()
        state = next_state
        total_reward += reward

    if episode % 1000 == 0:
        target_model.load_state_dict(model.state_dict())

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Đánh giá mô hình
def evaluate_model(env, model, num_episodes=EVAL_EPISODES):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward: {avg_reward}")
    return avg_reward

evaluate_model(env, model)