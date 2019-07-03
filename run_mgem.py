"""
# --------------------------------------------------------------------
# | 1. Overview                                                      |
# --------------------------------------------------------------------
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

A brief formulation of our problem is given as follows:
    --> Given C_k, P_k, SOC_(k-1)
    --> Find u_k(action of the battery)
    --> So that {u_k} yields minimum consumption of diesel

BTW, as for the stochastic setting, out formulation is different:
    --> Given [C_(k-hc), ..., C_(k-1), P_(k-hp), ..., P_(k-1), SOC_(k-1)]
    --> Find u_k
    --> So that {u_k} yields minimum consumption of diesel

In the light of the statement above, we have several solutions/tools in
our disposition.
    1. Rule-based control
    2. Dynamic programming
    3. Reinforcement learning(lqa, Q-learning, DQN, REINFORCE, DPG, DDPG, Actor-Critic...)

Not every method mentioned above can be applied to both settings.
Due to curse of dimension, namely the exponentially increasing
computation cost from more discrete or discretization-of state space,
observation space and the action space, applying certain methods, like
tabular ones, to the second setting may not be realistic. Thus it
requires discretion to choose the proper method according to characteristics
of the problem.


# --------------------------------------------------------------------
# | 2. Plan for deterministic setting                                |
# --------------------------------------------------------------------
In deterministic setting, we have SOC and time series position as the
state representation. From the benchmark work which is realized with
rule based control and dynamic programming techniques, we see that dynamic
programming can outperform simple rule based control. So the remaining work
is to use reinforcement learning methods to solve this deterministic optimisation
problem. The methods we shall test here are
    1. Rule-based control, /_\
    2. Dynamic programming, /_\
    3. lfa,
    4. Q-learning,
    5. DQN,
    6. REINFORCE,
    7. DDPG.

We are expecting the result to be useful in the following two aspects:
    1. We shall have all five methods mentioned above giving constructive
        results, meaning they can effectively optimize the energy management
        problem.Yet their performance are all bounded by the DP result. And
        between these five methods themselves, we may predict that DQN <= lfa in
        terms of performance and DDPG >= DQN in terms of convergence speed et etc.
        It might be interesting to look at the differences between the prediction from
        a theoretical point of view and practical implementation results.
    2. We shall be able to compare the models from the deterministic setting with
        the models from the stochastic setting. This can be done in two ways.
        --> We can use the result from the the deterministic setting as benchmark work
            which we shall compare with the result of models from stochastic setting.
            Thus we shall be able to observe the generalization ability of our models,
            which characterize the ability to tackle environmental uncertainties. And this
            is our ultimate goal of using reinforcement methods, to empower the model with
            the ability to deal with new situations.
        --> We can optimize the total time series data(ex, one year) by DP, and then compare
            it with stochastic models.
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

a = np.array([1, 2, 3])
b = np.array([1,2,3])
print(np.dot(a,b))