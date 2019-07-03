"""
Tabular method: Sarsa
"""
import numpy as np
import copy
import time

from rlmgem.env.mgem_env import DefineEnv


env = DefineEnv()
soc_space = np.arange(0.2,1.01,0.1)
action_space = np.arange(-0.05,0.06,0.01)
action_dim = len(action_space)
soc_dim = len(soc_space)

def reset():
    Q = np.zeros((144,soc_dim,action_dim))
    NSA = np.zeros((144,soc_dim,action_dim))
    return Q, NSA


