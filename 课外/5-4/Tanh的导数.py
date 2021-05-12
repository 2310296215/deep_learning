import numpy as np


def sigmoid(x):
    return 1 / (1 - np.exp(-1))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def dect(x):
    return 1 - tanh(x) ** 2
