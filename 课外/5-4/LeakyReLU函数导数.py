import numpy as np


# 其中 p 为 LeakyReLU 的负半段斜率，为超参数
def dect(x, p):
    dx = np.ones_like(x)
    dx[x < 0] = p
    return dx


print(dect(-2, 2))
