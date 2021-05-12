import numpy as np


def dect(x):  # ReLU函数的导数
    d = np.array(x, copy=True)  # 用于保存梯度的张量
    d[x < 0] = 0
    d[x >= 0] = 1
    return d


print(dect(0))
