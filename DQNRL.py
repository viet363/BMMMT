import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.005
NUM_EPISODES = 200
EVAL_INTERVAL = 5
TAU = 0.01
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 500000


CSV_FILE = "kdd_test1.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_and_preprocess_data(csv_file=CSV_FILE):

    df = pd.read_csv(csv_file)


    df['label_binary'] = df['labels'].apply(lambda x: 0 if x == 'normal' else 1)

    df['protocol_type'] = df['protocol_type'].astype('category').cat.codes
    df['service'] = df['service'].astype('category').cat.codes
    df['flag'] = df['flag'].astype('category').cat.codes

    features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
    X = df[features].values.astype(np.float32)
    y = df['label_binary'].values.astype(np.int64)

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min + 1e-8
    X_scaled = (X - X_min) / X_range

    return X_scaled, y



class NetworkIntrusionEnv(gym.Env):
    def __init__(self, X_data, y_data):
        super(NetworkIntrusionEnv, self).__init__()
        self.X_data = X_data  # (num_samples, num_features)
        self.y_data = y_data  # (num_samples,)
        self.num_samples = len(self.X_data)
        self.state_size = self.X_data.shape[1]

        self.action_size = 2

        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.state_size,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)

        self.current_idx = 0
        self.state = None

    def reset(self):
        self.current_idx = 0
        self.state = self.X_data[self.current_idx]
        return self.state

    def step(self, action):
        label = self.y_data[self.current_idx]
        # Phần thưởng: +1 nếu dự đoán đúng, -1 nếu sai
        reward = 1 if (action == label) else -1

        self.current_idx += 1
        done = (self.current_idx >= self.num_samples)

        if not done:
            next_state = self.X_data[self.current_idx]
        else:
            next_state = self.state

        self.state = next_state
        return next_state, reward, done, {}



class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.reset_parameters()
        self.reset_noise(std_init)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.5 / np.sqrt(self.weight_mu.size(1)))

    def reset_noise(self, std_init=0.5):
        self.weight_epsilon.normal_(0, std_init)

    def forward(self, x):
        return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon)

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

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def evaluate_model(env, model, num_episodes=5):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).to(device))
                action = q_values.argmax().item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"[EVAL] Avg Reward: {avg_reward:.2f}, Std: {std_reward:.2f}")
    return avg_reward



def main():
    X_scaled, y = load_and_preprocess_data(CSV_FILE)

    env = NetworkIntrusionEnv(X_scaled, y)

    state_size = env.state_size
    action_size = env.action_size

    model = DuelingDQN(state_size, action_size).to(device)
    target_model = DuelingDQN(state_size, action_size).to(device)
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
                    q_values = model(torch.FloatTensor(state).to(device))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

        if len(memory) > BATCH_SIZE:
            loss = train_model(model, target_model, optimizer, memory)
            writer.add_scalar("Loss", loss, episode)

        # Soft update
        soft_update(target_model, model, TAU)

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        writer.add_scalar("Total Reward", total_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)

        if episode % EVAL_INTERVAL == 0:
            avg_reward = evaluate_model(env, model)
            writer.add_scalar("Avg Reward", avg_reward, episode)
            print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, Avg Eval: {avg_reward:.2f}")

    final_avg = evaluate_model(env, model)
    print(f"Final Avg Reward: {final_avg:.2f}")

    torch.save(model.state_dict(), "dueling_dqn_model.pth")
    print("Model saved to dueling_dqn_model.pth")


if __name__ == "__main__":
    main()
