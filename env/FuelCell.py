import numpy as np

class FuelCell:
    def __init__(self, consumption_function, max_power, min_power):
        self.consumption = consumption_function
        self.max_power = max_power
        self.min_power = min_power


    def fuelcell_action_space(self):
        return [self.min_power, self.max_power]
