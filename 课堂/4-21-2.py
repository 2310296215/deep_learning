import numpy as np  # 引入数值计算的包
from mpl_toolkits.mplot3d import Axes3D # 引入画3D三维图像的包
import matplotlib as mpl
import matplotlib.pyplot as plt  # 引入画图的包
from sympy import *  # 引入求导数的包

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 定义字体微软雅黑
mpl.rcParams['axes.unicode_minus'] = False  # 使得函数图像坐标轴上的负坐标的负号能正常显示

x = np.arange(-5, 5, 0.25)  # 定义X坐标轴范围
y = np.arange(-5, 5, 0.25)  # 定义Y坐标轴范围
x, y = np.meshgrid(x, y)  # 定义网络
print(x)
print(y)
f = []
for i in range(3, 92):
    f.append('(' + str(i) + '*' + 'x' + '+' + 'y' + '-' + str(i + 1) + ')' + '**2')
fun = '+'.join(i for i in f)
res = eval(fun)
ax = plt.figure(figsize=(10, 5), facecolor='w').gca(projection='3d')  # 定义三维图像大小和背景颜色白色
ax.plot_surface(x, y, res, rstride=1, cstride=1, cmap=plt.get_cmap('gray_r'), linewidth=0, antialiased=False)  # 定义图像外观
plt.xlabel(u'X', fontsize=16)  # 定义X坐标轴字体大小
plt.ylabel(u'Y', fontsize=16)  # 定义Y坐标轴字体大小
plt.title(u'距离曲面', fontsize=18)  # 定义标题文字和字体大小
plt.show()  # 画距离函数图像

x = Symbol('x')
y = Symbol('y')
res = eval(fun)
result = solve([diff(res, x), diff(res, y)], [x, y])  # 求解对x,y偏导均为0的解
coef1 = result[x].evalf()  # 提取系数
coef2 = result[y].evalf()  # 提取系数
print(coef1)  # 打印系数
print(coef2)  # 打印系数

x = np.linspace(0, 100, num=500000)  # 0到10之间取500000个点，使图像连贯
y = coef1 * x + coef2  # 定义拟合的曲线的函数
plt.figure(figsize=(10, 5), facecolor='w')  # 定义图像大小和背景颜色白色
plt.plot(x, y, 'y.', markersize=1)  # 用黄色画这500000个点
plt.plot(np.arange(3, 92), np.arange(4, 93), 'r.', markersize=8)
plt.xlabel(u'X', fontsize=16)  # 定义X坐标轴字体大小
plt.ylabel(u'Y', fontsize=16)  # 定义Y坐标轴字体大小
plt.title(u'拟合直线', fontsize=18)  # 定义标题文字和字体大小
plt.show()  # 画拟合直线图像
