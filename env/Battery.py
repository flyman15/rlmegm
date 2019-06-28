import numpy as np


class Battery:
    def __init__(self, efficiency_function, dynamics_function, max_power, min_power, initial_state, action_space=[-0.05, 0.05]):
        self.efficiency_function = efficiency_function
        self.max_power = max_power
        self.min_power = min_power
        self.external_dynamics = dynamics_function
        self.actions = action_space
        # state of charge
        self.init_soc = initial_state
        self.soc = initial_state

    def initBat(self):
        self.soc = self.init_soc

    def dynamics(self, time_step, battery_action):
        soc, soc_process = self.external_dynamics(self.soc, self.efficiency_function, battery_action, time_step)
        self.soc = soc
        return self.soc, soc_process

    def action_space(self):
        """
        Return the possible rate of change of the SOC of the battery
        """
        return self.actions
