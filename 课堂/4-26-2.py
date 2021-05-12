import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import *

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
x = np.linspace(-15, 15, num=50000)
y = 255341*x**2 - 519048*x + 263796
plt.figure(figsize=(10, 5), facecolor='w')
plt.plot(x, y, 'y.', markersize=1)
a = 15
b = 255341*a**2 - 519048*a + 263796
plt.plot(a, b, 'g.', markersize=8)
learning_rate = 0.000001
for _ in range(10):
    c = a
    a = Symbol('a')
    a = c - learning_rate * (diff(255341*a**2 - 519048*a + 263796, a).subs('a', c))
    b = 255341*a**2 - 519048*a + 263796
    if _ == 9:
        plt.plot(a, b, 'r', markersize=8)
        print(a)
        print(c)
        print(37. / 28)
    else:
        plt.plot(a, b, 'g', markersize=8)
plt.xlabel(u'X', fontsize=16)
plt.ylabel(u'Y', fontsize=16)
plt.title(u'梯度下降法找最小值点', fontsize=18)
plt.show()
x = np.linspace(0, 10, num=50000)
y = a * x
plt.figure(figsize=(10, 5), facecolor='w')
plt.plot(x, y, 'y.', markersize=1)
plt.plot(np.arange(3, 92), np.arange(4, 93), 'r.', markersize=8)
plt.xlabel(u'X', fontsize=16)
plt.ylabel(u'Y', fontsize=16)
plt.title(u'拟合直线', fontsize=18)
plt.show()
