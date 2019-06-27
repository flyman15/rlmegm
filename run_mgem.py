"""
This part of scripts contains code for optimising the problem of
energy management in the setting of deterministic energy system, which
is composed of PV, Diesel generator, battery and load, in a micro-grid
which is not connected with any other grid.

This system is called as deterministic because we only deal with a time
series of length equal to 144. The goal of optimisation is to select
the best action series through the time to consume minimum amount of diesel.
All fluctuations in solar radiation and in energy demand with be encoded
inside the natural time series order instead of as independent variables
in the state vector representation.

In the light of the statement above, we have several solutions/tools in
our disposition.
    1. Rule-based control
    2. Dynamic programming
    3. Reinforcement learning(lqa, Q-learning, DQN, REINFORCE, DPG, DDPG, Actor-Critic)

A brief formulation of our problem is given as follows:
    --> Given C_k, P_k, SOC_(k-1)
    --> Find u_k(action of the battery)
    --> So that {u_k} yields minimum consumption of diesel

BTW, as for the stochastic setting, out formulation is completely different:
    --> Given [C_(k-hc), ..., C_(k-1), P_(k-hp), ..., P_(k-1), SOC_(k-1)]
    --> Find u_k
    --> So that {u_k} yields minimum consumption of diesel

Not every method mentioned above can be applied to both settings.
Due to curse of dimension, namely the exponentially increasing
computation cost from more discrete or discretion-of state space,
observation space and the action space, applying certain methods, like
tabular ones, to the second setting may not be realistic. Thus it
requires discretion to choose the proper method.
"""
import numpy as np
import math
from rlmgem.env.mgem_env import ElectricityGenerator, Environment
from rlmgem.utils.plotting import plot_profile


class Defaults:
    """
    Class used to set up all parameters for
    the algorithm and the environment.
    """
    # ------------------------
    # Environment parameters
    # ------------------------
    TIME_SPAN = 144     # 24*6 / 10min
    TIME_STEP = 1 / 6   # 1/6h

    # ------------------------
    # Experiment parameters
    # ------------------------
    NUMBER_EPISODE = int(1e3)

    # ------------------------
    # Agent parameters
    # ------------------------
    STEP_SIZE = 0.01
    EXPLORATION_PROBABILITY = 0.05
    DISCOUNT_FACTOR = 0.99
    TRACE_DECAY_RATE_LIST = list(np.arange(0,11)/10)



'''Define an example of ideal load and pv energy profile'''
load_profile = 20*np.ones(Defaults.TIME_SPAN)
energy_gene_profile = 30*np.ones(Defaults.TIME_SPAN)
energy_gene_profile[0:int(Defaults.TIME_SPAN/4)] = 8
energy_gene_profile[int(Defaults.TIME_SPAN*2/3):Defaults.TIME_SPAN] = 5
# profiles = dict()
# profiles['load'] = load_profile
# profiles['energy generation'] = energy_gene_profile
# plot_profile(profiles)
action_space = np.arange(-0.05,0.05,0.005) # limit rate of change of SOC each step
