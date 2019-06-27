"""
naive rule-based control strategy ???
Maybe the optimal policy already
"""

import numpy as np
import copy
import matplotlib.pyplot as plt

from rlmgem.env.CleanPowerGenerator import ElectricityGenerator
from rlmgem.env.Battery import Battery
from rlmgem.env.FuelCell import FuelCell
from rlmgem.env.mgem_env import Environment


def battery_efficiency(output_power):
    return 0.9

def battery_dynamics(soc, efficiency_function, output_power, time_step = 1/6):
    n_bat = 5
    energy_bat_max = 4.920*n_bat
    sigma_bat = 0
    # min_soc = 0.2
    # max_soc = 1
    new_soc = soc*(1-sigma_bat) + output_power*time_step*efficiency_function(output_power)/energy_bat_max
    return new_soc, [new_soc]

def fuelcell_consumption(power_output, time_step):
    nominal_power = 30
    return time_step*(0.04667 * nominal_power + 0.26267 * power_output)

def PV_prod(env_profile, time_posi):
    area_pv = 1.94
    efficiency_pv = 0.206
    N_pv = 81
    P_pv = env_profile[time_posi]*area_pv*N_pv*efficiency_pv
    return P_pv


load = np.load('C:/Users\lenovo\Documents\PycharmProjets\Easy21/rlmgem/data/load.npy')
pv_prod_profile = np.load('C:/Users\lenovo\Documents\PycharmProjets\Easy21/rlmgem/data/PV_prod.npy')
time_span = len(load)

# --------------------Test------------------------
# time_span = 20
# load = 20*np.ones(time_span)
# pv_prod_profile = 0.3*np.ones(time_span)
# pv_prod_profile[7:15] = 1
# -----------------------------------------------------------------------

TIME_STEP = 1/6
myBattery = Battery(efficiency_function=battery_efficiency, dynamics_function=battery_dynamics,
                    max_power=60, min_power=-60, initial_state=0.8)
myFuelcell = FuelCell(consumption_function=fuelcell_consumption, max_power=37.5, min_power=0)
myPV = ElectricityGenerator(prod_profile=pv_prod_profile, profile_type='radiation', prod_function=PV_prod)
myEnv = Environment(load_profile=load, battery=myBattery, fuelcell=myFuelcell, WT=None, PV=myPV, time_step=TIME_STEP)

plt.style.use('bmh')
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(121)
# ax1.plot(load)
# ax1.title.set_text('Load')
# ax2 = fig1.add_subplot(122)
# ax2.plot(pv_prod_profile)
# ax2.title.set_text('PV production')
# plt.show()

'''battery parameters'''
n_bat = 5
energy_bat_max = 4.920 * n_bat
sigma_bat = 0
max_soc = 1
min_soc = 0.2

'''key profiles'''
P_pv_real = np.zeros(len(load))
print(P_pv_real)
for e in range(0, len(pv_prod_profile)):
    P_pv_real[e] = PV_prod(pv_prod_profile, e)
P_pv = copy.copy(P_pv_real)
battery_output_power = np.zeros(len(load))
fuel_output_power = np.zeros(len(load))
fuel_consume = np.zeros(len(load))
P_gene = load

soc_process = list()

while myEnv.time_posi < len(load):
    soc_process.append(myEnv.battery.soc)
    k = myEnv.time_posi
    # calculate the maximum charging power
    P_ch_max = 0.279*6*n_bat*myEnv.battery.soc
    # calculate the minimum charging power
    P_ch_min = -0.279*6*n_bat*myEnv.battery.soc
    Delta_P = myEnv.clean_power_generation() - myEnv.load()
    if Delta_P > 0:
        if myEnv.battery.soc < max_soc:
            if Delta_P < P_ch_max:
                battery_output_power[k] = Delta_P
                myEnv.step_transition(Delta_P/myEnv.battery.max_power)
            else:
                battery_output_power[k] = P_ch_max
                P_pv_real[k] = myEnv.load() + P_ch_max
                myEnv.step_transition(P_ch_max/myEnv.battery.max_power)
        else:
            battery_output_power[k] = 0
            myEnv.step_transition(0)
    else:
        if myEnv.battery.soc > min_soc:
            if Delta_P > P_ch_min:
                battery_output_power[k] = Delta_P
                myEnv.step_transition(Delta_P/myEnv.battery.max_power)
            else:
                battery_output_power[k] = P_ch_min
                myEnv.step_transition(P_ch_min/myEnv.battery.max_power)
                if -(Delta_P - P_ch_min) < myEnv.fuelcell.max_power:
                    fuel_output_power[k] = -(Delta_P - P_ch_min)
                    fuel_consume[k] = myEnv.fuelcell.consumption(fuel_output_power[k], TIME_STEP)
                else:
                    fuel_output_power[k] = myEnv.fuelcell.max_power
                    fuel_consume[k] = myEnv.fuelcell.consumption(fuel_output_power[k])
                    P_gene[k] = fuel_output_power[k] + P_pv[k] - battery_output_power[k]
        else:
            myEnv.step_transition(0)
            battery_output_power[k] = 0
            if -(Delta_P) < myEnv.fuelcell.max_power:
                fuel_output_power[k] = -Delta_P
                fuel_consume[k] = myEnv.fuelcell.consumption(fuel_output_power[k], TIME_STEP)
            else:
                fuel_output_power[k] = myEnv.fuelcell.max_power
                P_gene[k] = fuel_output_power[k] + P_pv[k]
                fuel_consume[k] = myEnv.fuelcell.consumption(fuel_output_power[k], TIME_STEP)


np.save('rules_diesel.npy', fuel_output_power)

plt.figure()
plt.plot(load, label = 'load')
plt.plot(P_pv, label = 'PV')
plt.plot(fuel_output_power, label = 'Diesel')
plt.plot(-1*(battery_output_power), label = 'Battery')
plt.legend()
plt.show()
plt.figure()
plt.plot(P_pv)
plt.show()
print(sum(fuel_consume))
plt.figure()
plt.plot(battery_output_power)
plt.show()
plt.figure()
plt.plot(soc_process)
plt.show()
# assert (P_pv_real == P_pv).all()
