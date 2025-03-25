import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.005
NUM_EPISODES = 1000
EVAL_INTERVAL = 10
TAU = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 500000
CSV_FILE = "kdd_test.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log file path
LOG_FILE = "training_log.txt"


def load_and_preprocess_data(csv_file=CSV_FILE):
    df = pd.read_csv(csv_file)
    df['label_binary'] = df['labels'].apply(lambda x: 0 if x == 'normal' else 1)
    df['protocol_type'] = df['protocol_type'].astype('category').cat.codes
    df['service'] = df['service'].astype('category').cat.codes
    df['flag'] = df['flag'].astype('category').cat.codes

    features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
    X = df[features].values.astype(np.float32)
    y = df['label_binary'].values.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


class NetworkIntrusionEnv(gym.Env):
    def __init__(self, X_data, y_data):
        super().__init__()
        self.X_data = X_data
        self.y_data = y_data
        self.num_samples = len(self.X_data)
        self.state_size = self.X_data.shape[1]
        self.action_size = 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)
        self.current_idx = 0
        self.state = None

    def reset(self):
        self.current_idx = 0
        self.state = self.X_data[self.current_idx]
        return self.state

    def step(self, action):
        reward = 1 if (action == self.y_data[self.current_idx]) else -1
        self.current_idx += 1
        done = (self.current_idx >= self.num_samples)
        if not done:
            self.state = self.X_data[self.current_idx]
        return self.state, reward, done, {}


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch Normalization
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, x):
        if x.dim() == 1:  # Xử lý batch size = 1
            x = x.unsqueeze(0)
        if x.size(0) == 1:  # Nếu batch size là 1, bỏ qua BatchNorm
            x = torch.relu(self.fc1(x))
        else:
            x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


def train_model(model, target_model, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return 0
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # Gradient Clipping
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
                action = model(torch.FloatTensor(state).to(device)).argmax().item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


def evaluate_model_on_test_data(model, X_test, y_test):
    model.eval()
    y_pred = []
    for state in X_test:
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        # Gọi hàm phản hồi an ninh: nếu dự đoán là tấn công (1) thì thực hiện chặn
        take_security_action(action, state)
        y_pred.append(action)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    return y_pred, accuracy, precision, recall, f1


# Hàm thực hiện hành động bảo mật: chặn nếu phát hiện tấn công, bỏ qua nếu bình thường.
def take_security_action(action, state):
    if action == 1:
        # Giả lập hành động chặn: in ra thông báo và ghi log.
        message = f"Attack detected at state {state}. Blocking connection."
        write_log(message)
    else:
        # Bình thường, không thực hiện hành động chặn.
        write_log("Normal traffic: no blocking action taken.")


def write_log(message):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(message + "\n")
    print(message)


# Hàm vẽ ma trận nhầm lẫn
def plot_confusion_matrix(y_true, y_pred, labels=["Normal", "Anomaly"]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


# Hàm vẽ biểu đồ Learning Curve
def plot_learning_curve(train_rewards, eval_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(train_rewards, label='Training Rewards')
    plt.plot(np.linspace(0, NUM_EPISODES, len(eval_rewards)), eval_rewards, label='Evaluation Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


def main():
    # Khởi tạo log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    write_log("Starting training...")

    X_scaled, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    env = NetworkIntrusionEnv(X_train, y_train)

    policy_net = DuelingDQN(env.state_size, env.action_size).to(device)
    target_net = DuelingDQN(env.state_size, env.action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=BUFFER_SIZE)

    epsilon = EPSILON_START
    train_rewards = []
    eval_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample() if np.random.rand() < epsilon else policy_net(
                torch.FloatTensor(state).to(device)).argmax().item()
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

        train_rewards.append(total_reward)

        if len(memory) > BATCH_SIZE:
            loss = train_model(policy_net, target_net, optimizer, memory)

        soft_update(target_net, policy_net, TAU)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % EVAL_INTERVAL == 0:
            avg_reward = evaluate_model(env, policy_net)
            eval_rewards.append(avg_reward)
            log_message = f"Episode {episode}, Avg Eval Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Reward: {total_reward:.4f}"
            write_log(log_message)

    write_log("Training complete. Evaluating on test data...")
    y_pred, accuracy, precision, recall, f1 = evaluate_model_on_test_data(policy_net, X_test, y_test)
    write_log(
        f"Test Data Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    plot_confusion_matrix(y_test, y_pred, labels=["Normal", "Anomaly"])
    plot_learning_curve(train_rewards, eval_rewards)


if __name__ == "__main__":
    main()
