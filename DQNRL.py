import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter



# Constants
STATE_SIZE = 10
ACTION_SIZE = 4
BUFFER_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.005
NUM_EPISODES = 4000
EVAL_INTERVAL = 100
TAU = 0.01
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Device configuration
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
        self.state = np.random.rand(self.state_size).astype(np.float32)
        return self.state

    def step(self, action):
        reward = 1 if (action == 1 and self.state[0] > 0.85) else -1 if action == 1 else 0
        self.state = np.random.rand(self.state_size).astype(np.float32)
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


# Dueling DQN Model
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = NoisyLinear(state_size, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


# Training Function
def train_model(model, target_model, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return 0
    batch = random.sample(memory, BATCH_SIZE)
    batch = np.array(batch, dtype=object)
    states = np.vstack(batch[:, 0])
    actions = np.array(batch[:, 1], dtype=np.int64).reshape(-1, 1)
    rewards = np.array(batch[:, 2], dtype=np.float32)
    next_states = np.vstack(batch[:, 3])
    dones = np.array(batch[:, 4], dtype=np.float32)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    with torch.no_grad():
        best_actions = model(next_states).argmax(dim=1, keepdim=True)
        next_q = target_model(next_states).gather(1, best_actions).squeeze(1)
        target_q = rewards + GAMMA * next_q * (1 - dones)

    current_q = model(states).gather(1, actions).squeeze(1)
    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Soft Update Function
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# Evaluation Function
def evaluate_model(env, model, num_episodes=20):
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
    std_reward = np.std(total_rewards)
    print(f"Average Reward: {avg_reward}, Std Reward: {std_reward}")
    return avg_reward


if __name__ == "__main__":
    env = NetworkIntrusionEnv()
    model = DuelingDQN(STATE_SIZE, ACTION_SIZE).to(device)
    target_model = DuelingDQN(STATE_SIZE, ACTION_SIZE).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    memory = deque(maxlen=BUFFER_SIZE)
    writer = SummaryWriter()

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
            state = next_state
            total_reward += reward

        if len(memory) > BATCH_SIZE:
            loss = train_model(model, target_model, optimizer, memory)
            writer.add_scalar("Loss", loss, episode)

        soft_update(target_model, model, TAU)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        writer.add_scalar("Total Reward", total_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)

        if episode % EVAL_INTERVAL == 0:
            avg_reward = evaluate_model(env, model)
            writer.add_scalar("Avg Reward", avg_reward, episode)
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Avg Reward: {avg_reward}")

    evaluate_model(env, model)
    torch.save(model.state_dict(), "dueling_dqn_model")
