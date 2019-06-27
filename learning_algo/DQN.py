"""
Using Deep-Q-learning to optimize the energy management problem in micro-grids
"""
import tensorflow as tf
import numpy as np
import copy
from keras.models import Model, Sequential
from keras.layers import Input, Layer, Dense, Flatten, concatenate, Activation, Conv2D, MaxPooling2D, Reshape, Permute

from rlmgem.utils.plotting import DQNPlot

class NN:
    """
    Deep Q-learning network using Keras

    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
        Set the random seed.
    action_as_input : Boolean
        Whether the action is given as input or as output
    """
    model = Sequential()



class MyQNetwork:
    def __init__(self, NeutralNetWork:NN):
        pass

