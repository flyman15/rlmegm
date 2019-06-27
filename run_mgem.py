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
