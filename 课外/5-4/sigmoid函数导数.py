import numpy as np


def Sigmoid(x):
    return 1 / (1 - np.exp(-x))


def dect(x):
    return Sigmoid(x) * (1 - Sigmoid(x))


print(dect(3))
