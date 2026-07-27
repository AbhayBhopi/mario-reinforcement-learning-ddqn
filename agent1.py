import itertools
import os
import random

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml

from dqn1 import DQN
from experience_reply1 import ReplayMemory

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(CURRENT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# AGENT CLASS 

class Agent:

    def __init__(self, param_set, state_dim, action_dim):

        self.param_set = param_set
        self.state_dim = state_dim
        self.action_dim = action_dim

        params_path = os.path.join(CURRENT_DIR, "parameters1.yaml")
        with open(params_path, "r") as f:
            all_params = yaml.safe_load(f)
            params = all_params[param_set]

        # Loading hyperparameters 
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon = self.epsilon_init

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.SmoothL1Loss()
        
        # Policy and target networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.memory = ReplayMemory(self.replay_memory_size)
        self.step_count = 0

        self.LOG_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.log"
        )

        self.MODEL_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.pt"
        )

    def act(self, state, train=True):
        """
        Choose action using epsilon-greedy policy.
        """
        if train and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_arr = np.asarray(state)
            state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)
            state_tensor = state_tensor / 255.0
            with torch.inference_mode():
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def learn(self):
        """
        Sample minibatch and perform a gradient descent update step.
        """
        if len(self.memory) < self.mini_batch_size:
            return None

        # Sample transitions: (state, action, reward, next_state, done)
        transitions = self.memory.sample(self.mini_batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for t in transitions:
            states.append(np.asarray(t[0]))
            actions.append(t[1])
            rewards.append(t[2])
            next_states.append(np.asarray(t[3]))
            dones.append(t[4])

        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device) / 255.0
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device) / 255.0
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        # Get current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Compute target Q values using Double DQN
        with torch.no_grad():
            # Action selection using policy network
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            # Action evaluation using target network
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        # Optimize policy network
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        # Network synchronization
        self.step_count += 1
        if self.step_count % self.network_sync_rate == 0:
            self.sync_networks()

        return loss.item()

    def sync_networks(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, self.MODEL_FILE)

    def load(self, model_file=None):
        file_to_load = model_file if model_file else self.MODEL_FILE
        if os.path.exists(file_to_load):
            checkpoint = torch.load(file_to_load, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            print(f"Loaded model checkpoint from {file_to_load}")
            return True
        else:
            print(f"No checkpoint found at {file_to_load}")
            return False