"""
A tentative to verify that reinforcement learning principles can be applied
in this situation.
"""
import numpy as np
import copy
import time
start_time = time.time()


from rlmgem.utils.plotting import LBFAPlot
from rlmgem.env.mgem_env import DefineEnv

env = DefineEnv()
state_dim = 144*8
state_feature = [[0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7],
                 [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]]
actions = np.arange(-0.05,0.06,0.01)
action_dim = len(actions)
# print(actions)

# step size
alpha = 0.01

# exploration probability
epsilon = 0.5

# Sarsa(lamda)
lamda = 0.1
episodes = int(10000)


def reset():
    # theta = np.random.rand(state_dim*action_dim,1)
    theta = np.zeros((state_dim*action_dim,1))
    return theta


def epsilonGreedy(soc, k):
    if np.random.random() < epsilon:
        # explore
        action = np.random.choice(actions)
    else:
        # exploit
        foo = [Q(soc, k, action)
                            for action in actions]
        action_index = np.argmax(foo)
        action = actions[action_index]
    return action


def features(soc, k, action):
    """
    Our state space is [SOC_k, k]
    """
    myfeature = np.zeros((8, 144, action_dim), dtype=np.int)
    one = np.int(1)

    for fi, e in enumerate(state_feature):
        if e[1]>=soc>=e[0]:
            myfeature[fi,:,:] += one
    myfeature[:,k,:] += one
    myfeature[:,:,np.where(actions==action)] += one
    myfeature[np.where(myfeature != np.int(3))] = 0
    myfeature[np.where(myfeature == np.int(3))] = one
    ww = myfeature.flatten()
    ww = ww.reshape([state_dim*action_dim,1])
    return ww


def Q(soc, k, action):
    ww = np.dot(np.transpose(features(soc, k, action)), theta)
    return ww[0]


def reconstruct(soc0):
    myenv = DefineEnv(soc0)

    soc_process = list()
    fuel_power_process = list()
    bat_action_process = list()
    bat_power_process = list()

    k = 0
    mycost = 0
    aPrime_index = np.argmax([Q(soc0, k, action) for action in actions])
    aPrime = actions[aPrime_index]
    Terminated = False

    soc_process.append(soc0)
    bat_action_process.append(aPrime)

    while True:
        socPrime, kPrime, r, Terminated, P_u, P_diesel = myenv.step(aPrime)
        fuel_power_process.append(-1*P_diesel)
        mycost += r
        bat_power_process.append(P_u)

        k += 1
        soc_process.append(socPrime)
        if Terminated:
            break
        foo = [Q(socPrime, k, action) for action in actions]
        aPrime_index = np.argmax(foo)
        aPrime = actions[aPrime_index]
        bat_action_process.append(aPrime)
    return mycost, soc_process, fuel_power_process, bat_action_process, bat_power_process


# -----------------------------------------------------------------------
# |Train                                                                |
# -----------------------------------------------------------------------
theta = reset()
cost_record = list()
for episode in range(episodes):

    Terminated = False
    E = np.zeros_like(theta) # Eligibility trace

    # initial state and action
    soc, k = env.initEnv()
    a = epsilonGreedy(soc, k)

    # sample environment
    while not Terminated:

        socPrime, kPrime, r, Terminated, _, _ = env.step(a)
        if not Terminated:
            aPrime= epsilonGreedy(socPrime, kPrime)
            tdError = r + Q(socPrime, kPrime, aPrime) - Q(soc, k, a)
        else:
            tdError = r - Q(soc, k, a)
        E = lamda*E + features(soc, k, a) # feature return must be same dimension as theta
        gradient = alpha*tdError*E
        theta += gradient

        if not Terminated:
            soc, k, a = socPrime, kPrime, aPrime
        # else:
        #     print('**********************', episode, '--------------------', k)


    if episode%100 == 0:
        print('**********************', episode, '--------------------', k)
        foo, _, _, _, _ = reconstruct(0.8)
        cost_record.append(foo)

print("--- %s seconds ---" % (time.time() - start_time))

# save theta to npy file
np.save('../figures/theta_lfa_10000_0.5.npy', theta)



# -----------------------------------------------------------------------
# |load the theta file                                                  |
# -----------------------------------------------------------------------
# theta = np.load('../figures/theta_lfa_10000_0.5.npy')


# -----------------------------------------------------------------------
# |Reconstruct the optimal path                                         |
# -----------------------------------------------------------------------
soc0 = 0.8
cost, soc_process, fuel_power_process, bat_action_process, bat_power_process = reconstruct(soc0)


# -----------------------------------------------------------------------
# |                         Print/Plot                                  |
# -----------------------------------------------------------------------
print(cost, '------------------------------------')
print(soc_process, '\n', bat_action_process,'\n', fuel_power_process)
# plot system power flow Vs time span
LBFAPlot.plot1(env, fuel_power_process, bat_power_process)
# plot battery control flow
LBFAPlot.plot2(soc_process)
# cost Vs. training number
LBFAPlot.plot3(cost_record)
