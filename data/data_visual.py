import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# -----------------------------------------------------
# '''
# ---------1. spotmarket_data_2007-2013.xls-----------
# a). Electricity consumption profile of one consumer
#     in the whole year of 2009, recorded 24h*365d.
#     Peak period: 09-20
#     Off-peak:    00-08, 21-24
#     Base:        Peak + Off-peak
# b). Electricity price of 2007~2013, recorded 24h*365d
#     Peak period: 09-20
#     Off-peak:    00-08, 21-24
#    Base:         mean(Peak, Off-peak)
# '''
#
# '''2. PV production profile'''
# BelgiumPV_prod_test = np.load('BelgiumPV_prod_test.npy')
# print(stats.describe(BelgiumPV_prod_test))
# print(BelgiumPV_prod_test.shape)
# BelgiumPV_prod_train = np.load('BelgiumPV_prod_train.npy')
# print(stats.describe(BelgiumPV_prod_train))
# print(BelgiumPV_prod_train.shape)
# print(24*365)
# plt.figure(1)
# plt.plot(BelgiumPV_prod_test[0:24*30])
# plt.show()
# plt.figure(2)
# plt.plot(BelgiumPV_prod_train[0:24*30])
# plt.show()
#
# '''3. Consumption profile'''
# consum_test = np.load('example_nondeterminist_cons_test.npy')
# consum_train = np.load('example_nondeterminist_cons_train.npy')
# print(consum_test.shape)
# print(consum_train.shape)
# plt.figure(1)
# plt.plot(consum_test[0:24*3])
# plt.show()
# plt.figure(2)
# plt.plot(consum_train[0:24*3])
# plt.show()
# ----------------------------------------------------------


# """Transform the data to [0, 1] scale"""
# load_data = pd.read_excel('Load.xlsx', index_col=None, header=None)
# print(load_data[2])
# load = np.array(load_data[2])/1000
# np.save('load.npy', load)
# print(load)

# pv_prod = pd.read_excel('PV_Daily.xlsx',index_col=0, header=None)
#
# print(pv_prod.iloc[1])
# pv_prod = np.array(pv_prod.iloc[1])
# np.save('PV_prod.npy', pv_prod)
# print(pv_prod)
plt.style.use('bmh')
load = np.load('load.npy')
pv_prod = np.load('PV_prod.npy')
plt.figure(1)
plt.plot(load)
plt.show()

plt.figure(2)
plt.plot(pv_prod)
plt.show()






