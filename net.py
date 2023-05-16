import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
# from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition Defintion
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Replay Memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x