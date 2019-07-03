import numpy as np
import sys
import copy
import math
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
        self.terminate = False

    def initEnv(self):
        """Reset the battery soc and the time series position"""
        self.battery.initBat()
        self.time_posi = 0
        self.terminate = False
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
        """Generic function in RL for environment dynamics
            :
            return -inf(since RL minimize the cost function) when the consequence state
            of the action is not feasible;
            return fuel consumed when P_diesel is not 0
            return 0 when P_diesel is 0

            ** Action taken is always related with state change rate thus it acts also directly
                on the state of the battery; But P_u must be obtained by considering the
                transform efficiency effect. It should be noted that when sign of P_u is different
                the efficiency definition is different.
        """

        """ Balanced power flow """
        # Self-defined inf
        MYINF = 60
        n_bat = 5
        energy_bat_max = 4.920 * n_bat
        x0 = self.battery.soc
        # calculate the maximum charging power
        P_ch_max = 0.279 * 6 * n_bat * x0
        # calculate the minimum charging power
        P_ch_min = -0.279 * 6 * n_bat * x0
        # calculate the charging power of the action taken
        P_u = action * x0 * energy_bat_max * self.battery.efficiency_function(action) / self.time_step

        if (P_ch_min >= P_u) or (P_u >= P_ch_max):
            end_of_episode = True
            cost = -1*MYINF
            P_diesel = 0
        else:
            P_load = self.load()
            P_cleanpower = self.clean_power_generation()
            Delta_P = P_cleanpower - P_load
            """Control and dynamic action for the battery and the fuel cell"""
            self.battery.dynamics(self.time_step, P_u)
            P_diesel = min(0, Delta_P-P_u)
            fuel_consumption = self.fuelcell.consumption(-1*P_diesel, self.time_step)
            self.time_posi += 1

            end_of_episode = self.is_terminate() or self.battery.isnot_legal()
            if self.battery.isnot_legal():
                cost = -1*MYINF
            else:
                cost = -1*fuel_consumption
        return self.battery.soc, self.time_posi, cost, end_of_episode, P_u, P_diesel

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

    def is_terminate(self):
        """
        Three types of terminate conditions:
            1. end of time series
            2. battery output power exceeds permitted value
            3. consequence state of battery action is not legal
        """
        if self.time_posi == len(self.load_profile):
            self.terminate = True
        return self.terminate


# -------------------------------------------------------------------------------------
"""Function must be defined as the format below to complete the Environment, Battery, 
   FuelCell, cleanPowerGenerator class"""
# def battery_efficiency(output_power):
#     return #efficiency as a scalar number
#
# def battery_dynamics(soc, efficiency_function, output_power, time_step = 1/6):
#     return # new_soc, [new_soc]
#
# def fuelcell_consumption(power_output, time_step):
#     return # fuel consumed as a scalar
#
# def PV_prod(env_profile, time_posi):
#     return # P_pv as a scalar
# -------------------------------------------------------------------------------------

def DefineEnv(soc_initial=None):
    def battery_efficiency(output_power):
        if output_power >= 0:
            return 0.9
        else:
            return 1.0/0.9

    def battery_dynamics(soc, efficiency_function, output_power, time_step=1 / 6):
        n_bat = 5
        energy_bat_max = 4.920 * n_bat
        sigma_bat = 0
        new_soc = soc * (1 - sigma_bat) + output_power * time_step * efficiency_function(output_power) / energy_bat_max
        return new_soc, [new_soc]

    def fuelcell_consumption(power_output, time_step=1 / 6):
        if power_output > 0:
            nominal_power = 30
            return time_step * (0.04667 * nominal_power + 0.26267 * power_output)
        else:
            return 0

    def PV_prod(env_profile, time_posi):
        area_pv = 1.94
        efficiency_pv = 0.206
        N_pv = 81
        P_pv = env_profile[time_posi] * area_pv * N_pv * efficiency_pv
        return P_pv

    load = np.load('C:/Users\lenovo\Documents\PycharmProjets\Easy21/rlmgem/data/load.npy')
    pv_prod_profile = np.load('C:/Users\lenovo\Documents\PycharmProjets\Easy21/rlmgem/data/PV_prod.npy')

    TIME_STEP = 1 / 6
    myBattery = Battery(efficiency_function=battery_efficiency, dynamics_function=battery_dynamics,
                        max_power=60, min_power=-60, initial_state=soc_initial)
    myFuelcell = FuelCell(consumption_function=fuelcell_consumption, max_power=37.5, min_power=0)
    myPV = ElectricityGenerator(prod_profile=pv_prod_profile, profile_type='radiation', prod_function=PV_prod)
    myEnv = Environment(load_profile=load, battery=myBattery, fuelcell=myFuelcell, WT=None, PV=myPV,
                        time_step=TIME_STEP)

    return myEnv # Environment class object


if __name__ == "__main__":
    pass
