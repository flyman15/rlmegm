import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

from rlmgem.env.CleanPowerGenerator import ElectricityGenerator
from rlmgem.env.Battery import Battery
from rlmgem.env.FuelCell import FuelCell


class Environment:
    """
    This environment class meets the needs for deterministic setting;
    Thus in this case, the state variable is [SOC_k, k]
    """
    def __init__(self, load_profile, battery: Battery, fuelcell: FuelCell,
                        WT: ElectricityGenerator = None, PV: ElectricityGenerator=None, time_step = 1/6):
        self.WT = copy.copy(WT)
        self.PV = copy.copy(PV)
        self.load_profile = load_profile
        self.battery = copy.copy(battery)
        self.fuelcell = copy.copy(fuelcell)
        self.time_posi = 0
        self.time_step = time_step

    def initEnv(self):
        """Reset the battery soc and the time series position"""
        self.battery.initBat()
        self.time_posi = 0
        return self.state()

    def step_transition(self, action):
        """Only used in the rule-based control"""

        """ Balanced power flow """
        battery_output_power = self.battery.max_power*action
        fuelcell_output_power = self.clean_power_generation() + battery_output_power - self.load()

        """Control and dynamic action for the battery and the fuel cell"""
        self.battery.dynamics(self.time_step, battery_output_power)
        fuel_consumption = self.fuelcell.consumption(fuelcell_output_power, self.time_step)

        self.time_posi += 1
        return battery_output_power, fuelcell_output_power, fuel_consumption

    def step(self, action):
        """ Balanced power flow """
        battery_output_power = self.battery.max_power * action
        fuelcell_output_power = self.clean_power_generation() + battery_output_power - self.load()

        """Control and dynamic action for the battery and the fuel cell"""
        self.battery.dynamics(self.time_step, battery_output_power)
        fuel_consumption = self.fuelcell.consumption(fuelcell_output_power, self.time_step)

        self.time_posi += 1
        return self.load(), self.clean_power_generation(), self.battery.soc, fuel_consumption, self.terminated()

    def state(self):
        return self.battery.soc, self.time_posi

    def action_space(self):
        """
        Return the possible action space of controlling the battery discharging or charging
        """
        return self.battery.action_space()

    def clean_power_generation(self):
        """
        Consider only the generation of clean power which comes from PV or/and WT
        """
        if self.WT is None:
            WT_power = 0
        else:
            WT_power = self.WT.output_power(self.time_posi)
        if self.PV is None:
            PV_power = 0
        else:
            PV_power = self.PV.output_power(self.time_posi)
        return WT_power + PV_power

    def load(self):
        return self.load_profile[self.time_posi]

    def terminated(self):
        return self.time_posi == len(self.load_profile)


# -------------------------------------------------------------------------------------
"""Function must be defined as the format below to complete the Environment, Battery, 
   FuelCell, cleanPowerGenerator class"""
def battery_efficiency(output_power):
    return #efficiency as a scalar number

def battery_dynamics(soc, efficiency_function, output_power, time_step = 1/6):
    return # new_soc, [new_soc]

def fuelcell_consumption(power_output, time_step):
    return # fuel consumed as a scalar

def PV_prod(env_profile, time_posi):
    return # P_pv as a scalar
# -------------------------------------------------------------------------------------

def DefineEnv():
    return Environment() # Environment class


if __name__ == "__main__":
    pass
