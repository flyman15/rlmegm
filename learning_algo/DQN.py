"""
Using Deep-Q-learning to optimize the energy management problem in micro-grids
"""
import itertools
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from rlmgem.utils.plotting import DQNPlot


INPUT_SIZE = 3+3+1
OUTPUT_SIZE = 1


# -------------------------------------------------------
# | Network structure                                   |
# -------------------------------------------------------
class MyQNetwork(nn.Module):
    def __init__(self, hidden_size=10):
        super(MyQNetwork, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, s):
        """
        :param s: must be a Variable of shape (samples, INPUT_SIZE)
        :return: tensor of values
        """
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# training netwwork
qnet = MyQNetwork()
# target network
tnet = MyQNetwork()

# -------------------------------------------------------
# | Replay Buffer                                       |
# -------------------------------------------------------
class ReplayBuffer:
    def __init__(self, size):
        assert isinstance(size, int) and size > 0, "size must be positive integer"
        self.size = size
        self._storage = []
        self._i = 0

    def __len__(self):
        return len(self._storage)

    def push(self,x):
        if len(self._storage) < self.size:
            self._storage.append(x)
        else:
            self._storage[self._i] = x
        self._i = (self._i + 1) % self.size

    def sample(self, batch_size, replace=False):
        if replace:
            return random.choices(self._storage, k = batch_size)
        else:
            return random.sample(self, k=batch_size)

# -------------------------------------------------------
# | Explore Policy                                      |
# -------------------------------------------------------
def EpsilonGreedyPolicy(epsilon):
    def epsilon_greedy_policy(qvalues):
        if random.random() < epsilon:
            return random.randint



