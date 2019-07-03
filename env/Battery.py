import numpy as np


class Battery:
    def __init__(self, efficiency_function, dynamics_function, max_power, min_power, initial_state,
                 mic_soc=0.2, max_soc=1.0, action_space=[-0.05, 0.05]):
        self.efficiency_function = efficiency_function
        self.max_power = max_power
        self.min_power = min_power
        self.external_dynamics = dynamics_function
        self.actions = action_space
        # state of charge
        self.init_soc = initial_state
        self.soc = initial_state
        self.min_soc = mic_soc
        self.max_soc = max_soc

    def initBat(self):
        if self.init_soc is None:
            self.soc = np.random.uniform(0.2,0.8)
        else:
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

    def isnot_legal(self):
        if self.soc < self.min_soc or self.soc > self.max_soc:
            return True
        else:
            return False
