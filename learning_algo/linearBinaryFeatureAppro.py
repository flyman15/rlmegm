"""
Remains to do:
    1. Function feature(~)
    2. Function reconstruct(~)
    3. Function LBFAPlot.plot1(~), LBFAPlot.plot2(~)
"""
import numpy as np
import copy

from rlmgem.utils.plotting import LBFAPlot
from rlmgem.env.mgem_env import DefineEnv

env = DefineEnv()
state_dim = (5*5*5)*(5*5*5)*9
actions = np.arange(-0.05,0.06,0.01)
# print(actions)

# step size
alpha = 0.01

# exploration probability
epsilon = 0.05

# Sarsa(lamda)
lamda = 0.1

episodes = int(1e4)


def reset():
    theta = np.random.rand(state_dim*len(actions),1)
    return theta


def epsilonGreedy(consumption, production, soc):
    if np.random.random() < epsilon:
        # explore
        action = np.random.choice(actions)
    else:
        # exploit
        action = np.argmax([Q(consumption, production, soc, action)
                            for action in actions])
    return action


def features(consumption, production, soc, action):
    """
    How to represent the feature vector in the most efficient way ?
    Its a question...
    Possible solution:
        1. Using a (3+3+1+1) dimension tensor; and use finally 'flatten' method to turn it into a vector
        2. Using simply a (5+5+5+5+5+5+9+11) dimension vector, and then give value to a (5*5*5)*(5*5*5)*9*11
            dimension vector.
    Both are very time consuming and memory consuming.
    Maybe not a reasonable way to implement reinforcement learning.
    The problem comes from the fact that we have to discretize the state space and the action space.
    """
    return 0


def Q(consumption, production, soc, action):
    return np.dot(features(consumption, production, soc, action), theta)


def reconstruct(soc0):
    return 0# soc_process, fuel_consumption_process, fuel_power_process, bat_action_process, bat_power_process


theta = reset()
for episode in range(episodes):

    Terminated = False
    E = np.zeros_like(theta) # Eligibility trace

    # initial state and action
    consumption, production, soc = env.initEnv()
    a = epsilonGreedy(consumption, production, soc)

    # sample environment
    while not Terminated:

        consumptionPrime, productionPrime, socPrime, r, Terminated = env.step(a)
        if not Terminated:
            aPrime= epsilonGreedy(consumptionPrime, productionPrime, socPrime)
            tdError = r + Q(consumptionPrime, productionPrime, socPrime, aPrime) - Q(consumption, production,soc, a)
        else:
            tdError = r - Q(consumption, production, soc, a)

        E = lamda*E + features(consumption, production, soc, a) # feature return must be same dimension as theta
        gradient = alpha*tdError*E
        theta += gradient

        if not Terminated:
            consumption, production, soc, a = consumptionPrime, productionPrime, socPrime, aPrime


# reconstruct the optimal path
soc0 = 0.8
soc_process, fuel_consumption_process, \
fuel_power_process, bat_action_process, bat_power_process = reconstruct(soc0)

# plot system power flow Vs time span
LBFAPlot.plot1()
# plot battery control flow
LBFAPlot.plot2()
