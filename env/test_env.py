"""
Test for the environment
"""
import numpy as np
import matplotlib.pyplot as plt

from rlmgem.env.CleanPowerGenerator import ElectricityGenerator
from rlmgem.env.Battery import Battery
from rlmgem.env.FuelCell import FuelCell
from rlmgem.env.mgem_env import Environment


def fuelcell_consumption(power_output, time_step):
    """
    Return the efficiency of the power source;
    Model as given in formula (1.2);
    """
    power_output = power_output / 10
    p1 = -0.05613
    p2 = 0.6959
    p3 = -0.003055
    q1 = 0.4712
    efficiency = (p1 * power_output * power_output + p2 * power_output + p3) / (power_output + q1)
    assert efficiency != 0
    return time_step * power_output *10/ efficiency


def SC_efficiency(power_output):
    """
    Return the efficiency of the power source;
    """
    power_output = power_output / 10
    j1_des = -0.0001399
    j2_des = -0.0009613
    j3_des = 0.9931
    j1_ch = -0.0001283
    j2_ch = 0.0009874
    j3_ch = 0.9931
    if power_output >= 0:
        efficiency = j1_des * power_output * power_output + j2_des * power_output + j3_des
    else:
        efficiency = j1_ch * power_output * power_output + j2_ch * power_output + j3_ch
    return efficiency


def SC_dynamics(battery_soc, efficiency_function, battery_action, time_step):
    max_soc = 1
    min_soc = 0.5
    Capa = 81
    R0 = 8.6e-3
    Vcmax = 198
    Emax = 1600
    Emin = 400

    efficiency = efficiency_function(battery_action)
    assert efficiency != 0
    if battery_action >= 0:
        Psc_out = battery_action / efficiency
    else:
        Psc_out = battery_action * efficiency
    num_iter = 1 / time_step
    soc_process = list()
    e = 0
    soc = battery_soc
    while min_soc <= soc <= max_soc and e <= num_iter:
        # using discrete calculation
        # Possibly using continuous differential equation
        # --------------------------------------------------------------------
        e += 1
        soc = soc
        soc_process.append(soc)
        Voc = soc * Vcmax
        Isc_out = (Voc - np.sqrt(Voc * Voc - 4 * R0 * Psc_out * 1000)) / (2 * R0)
        soc +=  Isc_out * time_step * 3600 / (Vcmax * Capa)
        # --------------------------------------------------------------------
    if min_soc <= soc <= max_soc:
        return soc, soc_process
    else:
        return soc_process[-1], soc_process


TIME_STEP = 1/60
myBattery = Battery(efficiency_function=SC_efficiency, dynamics_function=SC_dynamics,
                    max_power=60, min_power=-60, initial_state=1)
myFuelcell = FuelCell(consumption_function=fuelcell_consumption, max_power=70, min_power=0)

# Test for fuelcell_efficiency function
plt.style.use('bmh')
plt.plot(np.linspace(0, 60, 100),
         [e*TIME_STEP / fuelcell_consumption(e, time_step=TIME_STEP)  for e in np.linspace(0, 60, 100)])
plt.xlabel('power/kW')
plt.ylabel('efficiency')
plt.title('Fuel Cell')
plt.show()

# Test for battery_efficiency function
plt.style.use('bmh')
plt.plot(np.linspace(-60, 60, 100), [SC_efficiency(e) for e in np.linspace(-60, 60, 100)])
plt.xlabel('power/kW')
plt.ylabel('efficiency')
plt.title('Battery: Converter part')
plt.show()

# Test for dynamics of the battery
soc = []
soc_final, soc_process = myBattery.dynamics(time_step = TIME_STEP, battery_action=-1.2)
print(soc_final, soc_process)
plt.style.use('bmh')
plt.plot(soc_process)
plt.xlabel('time/n*' + str(TIME_STEP*60) + 'min')
plt.ylabel('SOC')
plt.title('the discharging curve of the battery')
plt.show()

soc_final, soc_process = myBattery.dynamics(time_step = TIME_STEP, battery_action = +1.2)
print(soc_final, soc_process)
plt.style.use('bmh')
plt.plot(soc_process)
plt.xlabel('time/n*' + str(TIME_STEP*60) + 'min')
plt.ylabel('SOC')
plt.title('the charging curve of the battery')
plt.show()

# def PV_prod_function():
#     pass
#
# def WT_prod_function():
#     pass
#
# myPV = ElectricityGenerator(prod_profile=, profile_type='radiation', prod_function=PV_prod_function, nominal_power=)
# myWT = ElectricityGenerator(prod_profile=, profile_type='windspeed', prod_function=WT_prod_function, nominal_power=)
#
# myEnv = Environment(consumption_profile=, WT=myWT, PV=myPV, battery=myBattery, fuelcell=myFuelcell)