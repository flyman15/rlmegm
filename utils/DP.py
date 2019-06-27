"""
Benchmark result
using Dynamic Programming to obtain the optimal path/policy
"""

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
plt.style.use('bmh')


# -----------------------------------------------------------------------
# |                          System Settings                            |
# -----------------------------------------------------------------------
def battery_efficiency(output_power):
    return 0.9

def battery_dynamics(soc, efficiency_function, output_power, time_step = 1/6):
    n_bat = 5
    energy_bat_max = 4.920*n_bat
    sigma_bat = 0
    new_soc = soc*(1-sigma_bat) + output_power*time_step*efficiency_function(output_power)/energy_bat_max
    return new_soc, [new_soc]

def fuelcell_consumption(power_output, time_step):
    nominal_power = 30
    return time_step*(0.04667 * nominal_power + 0.26267 * power_output)#0.04667 0.26267

def PV_prod(env_profile, time_posi):
    area_pv = 1.94
    efficiency_pv = 0.206
    N_pv = 81
    P_pv = env_profile[time_posi]*area_pv*N_pv*efficiency_pv
    return P_pv

TIME_STEP = 1/6
load = np.load('C:/Users\lenovo\Documents\PycharmProjets\Easy21/rlmgem/data/load.npy')
pv_prod_profile = np.load('C:/Users\lenovo\Documents\PycharmProjets\Easy21/rlmgem/data/PV_prod.npy')
time_span = len(load)

# -----------------------------------------------------------------------
# time_span = 20
# load = 20*np.ones(time_span)
# pv_prod_profile = 0.3*np.ones(time_span)
# pv_prod_profile[7:15] = 1
# -----------------------------------------------------------------------

step_state = 0.001
action_space = np.arange(-0.05,0.05,0.005) # limit rate of change of SOC each step
state_space = np.arange(0.2,1.0001,step_state)

leng_action_space = len(action_space)
leng_state_space = len(state_space)


mu = np.zeros((leng_state_space, time_span))
# terminal-cost is zero in this case
Y = np.zeros((leng_state_space, time_span+1))


def error(u, x):
    min_soc = 0.2
    max_soc = 1
    if abs(u) <= 1e-10:
        return False
    if u < 0 and x<=min_soc:
        return True
    elif u > 0 and x>=max_soc:
        return True
    else:
        return False


def membership(x):
    p1 = np.sum((x>state_space))
    p2 = np.sum((x<state_space))
    # print(p1,p2)
    # assert (p1+p2)<= leng_state_space
    if (p1+p2) == leng_state_space:
        if p2==0:
            return p1-1, None, None
        elif p1==0:
            return 0,None,None
        else:
            return p1-1, p1, (state_space[p1]-x)/step_state
    else:
        return p1, None, None
    # which returns the ceiling value
    # posi = np.sum((x > state_space))
    # return min(int(posi), leng_state_space-1)


def state_transition(k, x0, u):
    bat_efficiency = 0.9
    n_bat = 5
    energy_bat_max = 4.920*n_bat

    # calculate the maximum charging power
    P_ch_max = 0.279*6*n_bat*x0
    # calculate the minimum charging power
    P_ch_min = -0.279*6*n_bat*x0
    # calculate the charging power of the action taken
    P_u = u*x0*energy_bat_max/(TIME_STEP*bat_efficiency)
    P_load_k = load[k]
    P_pv_k = PV_prod(pv_prod_profile, k)
    Delta_P = P_pv_k - P_load_k

    if (P_ch_min >= P_u) or (P_u >= P_ch_max) or error(u, x0):
        x = x0
        cost = math.inf
        P_diesel = 0
    else:
        x = x0 + x0*u
        if Delta_P - P_u >= 0:
            P_diesel = 0
            cost = 0
        else:
            P_diesel = Delta_P - P_u
            cost = fuelcell_consumption(-1*P_diesel, TIME_STEP)
    assert cost >= 0
    return cost, x, -1*P_diesel


# -----------------------------------------------------------------------
# |                            DP algorithm                             |
# -----------------------------------------------------------------------
info_mx = dict()
for i in range(0, time_span):
    # k in [0, 143] in the loop; k = 144 already implemented
    # for giving the terminal cost out of the loop
    L_k = np.zeros((leng_state_space, leng_action_space))
    X_k_plus_1 = np.zeros((leng_state_space, leng_action_space))
    Y_k = np.zeros((leng_state_space, leng_action_space))
    k = time_span-i-1

    for m in range(0, leng_state_space):
        for n in range(0, leng_action_space):
            L_k[m, n], X_k_plus_1[m, n], _= state_transition(k, state_space[m], action_space[n]) # FUNCTION WHICH GIVES THE STATE TRANSITION DYNAMICS
            a, b, c = membership(X_k_plus_1[m,n])
            if b is None:
                Y_k[m, n] = Y[a, k + 1] + L_k[m,n]
            else:
                Y_k[m, n] = (1-c)*Y[a, k + 1] + c*Y[b, k+1] + L_k[m, n]

        Y[m,k] = min(Y_k[m,:])
        mu[m,k] = int(np.argmin(Y_k[m,:]))
    info_mx[str(k)] = Y_k


# -----------------------------------------------------------------------
# |                  Reconstruct the optimal path                       |
# -----------------------------------------------------------------------
'''Method 1: without considering overall structure of the solution'''
# n_bat = 5
# energy_bat_max = 4.920*n_bat
# bat_efficiency = 0.9
#
# x0 = 0.8
# a, b,c = membership(x0)
# u0 = action_space[int(mu[a,0])]
# power_bat0 =u0*x0*energy_bat_max/(TIME_STEP*bat_efficiency)
#
# x = np.zeros(time_span)
# u = np.zeros(time_span)
# power_bat = np.zeros(time_span)
#
# power_bat[0] = power_bat0
# x[0] = x0
# u[0] = u0
#
# P_diesel = np.zeros(time_span)
# for i in range(1, time_span):
#     foo, x[i], P_diesel[i-1] = state_transition(i-1, x[i-1], u[i-1])
#     a, b, c = membership(x[i])
#     u[i] = action_space[int(mu[a, i])]
#     power_bat[i] = u[i]*x[i]*energy_bat_max/(TIME_STEP*bat_efficiency)
#
# i = time_span-1
# foo1, foo2, P_diesel[i] = state_transition(i, x[i], u[i])
#

'''Method 2: Considering the overall structure of the solution
             by penalizing the sharp change of output power in the 
             info_mx 
'''
def penalization(u, step = 10):
    penal_base = math.inf
    total = 0*action_space
    for i in range(1,min(step+1, len(u)+1)):
        u0 = u[-1*i]
        bias = abs(action_space-u0)
        bias[bias>=0.07]=penal_base
        bias[bias<0.07] = 0
        total += bias
    return total

def reconstruct(x0):
    n_bat = 5
    energy_bat_max = 4.920 * n_bat
    bat_efficiency = 0.9

    a, b, c = membership(x0)
    # u0 = action_space[int(mu[a, 0])]
    myaction = np.argmin(info_mx[str(0)][a,:])
    u0 = action_space[int(myaction)]
    power_bat0 = u0 * x0 * energy_bat_max / (TIME_STEP * bat_efficiency)

    x = np.zeros(time_span)
    u = np.zeros(time_span)
    power_bat = np.zeros(time_span)
    P_diesel = np.zeros(time_span)

    power_bat[0] = power_bat0
    x[0] = x0
    u[0] = u0
    penalization_step = 5

    for i in range(1, time_span):
        _, x[i], P_diesel[i - 1] = state_transition(i - 1, x[i - 1], u[i - 1])
        a, b, c = membership(x[i])
        penalized_info = np.add(info_mx[str(i)][a, :], penalization(u[0:i]))
        myaction = np.argmin(penalized_info)
        u[i] = action_space[int(myaction)]
        power_bat[i] = u[i] * x[i] * energy_bat_max / (TIME_STEP * bat_efficiency)

    i = time_span - 1
    _, _, P_diesel[i] = state_transition(i, x[i], u[i])
    return x, u, power_bat, P_diesel # x: path; u: action; power_bat: battery power output; P_diesel: power output of the diesel engine


x0 = 0.8
x, u, power_bat, P_diesel = reconstruct(x0)




# -----------------------------------------------------------------------
# |                      Calculate the cost                             |
# -----------------------------------------------------------------------
np.save('DP_diesel.npy', P_diesel)
rules_based_control_diesel = np.load('C:/Users\lenovo\Documents\PycharmProjets\Easy21/rlmgem/utils/rules_diesel.npy')
cost_rules = 0
mycost = 0
for e in range(0, time_span):
    cost_rules += fuelcell_consumption(rules_based_control_diesel[e], TIME_STEP)
    mycost += fuelcell_consumption(P_diesel[e], TIME_STEP)


# -----------------------------------------------------------------------
# |                         Print/Plot                                  |
# -----------------------------------------------------------------------
print(action_space)
print(state_space)
print(P_diesel.shape)
print('{', mycost, cost_rules, '}')

plt.figure()
plt.plot(load, label = 'load')
plt.plot([PV_prod(pv_prod_profile, e) for e in range(0, time_span)], label = 'PV')
plt.plot(P_diesel, label = 'Diesel')
plt.plot(-1*power_bat, label = 'Battery')
plt.legend()
plt.show()

plt.figure()
plt.plot(rules_based_control_diesel, label = 'rules-based control')
plt.plot(P_diesel, label = 'DP')
plt.title('contrast between DP and rules-based control')
plt.legend()
plt.show()

plt.figure()
plt.plot(x)
plt.xlabel('time span')
plt.ylabel('state')
plt.title('state transition')
plt.show()

plt.figure()
plt.plot(power_bat)
plt.xlabel('time span')
plt.ylabel('P')
plt.title('battery output')
plt.show()

plt.figure()
plt.plot(u)
plt.xlabel('time span')
plt.ylabel('action')
plt.title('battery action')
plt.show()

