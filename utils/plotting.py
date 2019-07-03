import matplotlib.pyplot as plt
import numpy as np
from rlmgem.env.mgem_env import Environment

def plot_profile(profiles):
    plt.style.use('bmh')
    plt.figure()
    for e in profiles:
        plt.plot(profiles[e], label = e)
    plt.legend()
    plt.show()


class LBFAPlot:
    @staticmethod
    def plot1(env:Environment, fuel_power_process, bat_power_process):
        load = env.load_profile
        time_span = len(load)

        plt.style.use('bmh')
        plt.figure()
        plt.plot(load, label='load')
        plt.plot([env.PV.output_power(e) for e in range(0, time_span)], label='PV')
        plt.plot(fuel_power_process, label='Diesel')
        plt.plot(-1 * np.array(bat_power_process), label='Battery')
        plt.title('LFA')
        plt.legend()
        # plt.savefig('../figures/DP.png', bbox_inches='tight')
        plt.show()


        plt.figure()
        plt.plot(np.array(fuel_power_process) + np.array(bat_power_process), label='supply')
        plt.plot(load, label='load')
        plt.legend()
        plt.title('power balance')
        plt.show()


    @staticmethod
    def plot2(soc_process):
        plt.style.use('bmh')
        plt.figure()
        plt.plot(soc_process)
        plt.title('SOC')
        plt.show()

    @staticmethod
    def plot3(costlist):
        plt.style.use('bmh')
        plt.figure()
        plt.plot(-1*np.array(costlist))
        plt.title('cost Vs. training episode')
        plt.xlabel('per 100 episode')
        plt.show()

class DQNPlot:
    @staticmethod
    def plot1():
        plt.style.use('bmh')
        pass

    @staticmethod
    def plot2():
        plt.style.use('bmh')
        pass


class DPGPlot:
    @staticmethod
    def plot1():
        plt.style.use('bmh')
        pass

    @staticmethod
    def plot2():
        plt.style.use('bmh')
        pass