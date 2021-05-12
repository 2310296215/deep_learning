import numpy as np  # 引入数值计算的包
import matplotlib as mpl
import matplotlib.pyplot as plt  # 引入画图的包
from sympy import *  # 引入求导数的包

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 定义字体微软雅黑
mpl.rcParams['axes.unicode_minus'] = False  # 使得函数图像坐标轴上的负坐标的负号能正常显示

x = np.linspace(-15, 15, num=500000)  # -15到15之间取500000个数据点，使图像连贯
f = []
for i in range(3, 92):
    f.append('(' + str(i) + '*' + 'x' + '-' + str(i + 1) + ')' + '**2')
y = eval('+'.join(i for i in f))
plt.figure(figsize=(10, 5), facecolor='w')  # 定义图像大小和背景颜色白色
plt.plot(x, y, 'y.', markersize=1)  # 用黄色画这500000个点
plt.xlabel(u'X', fontsize=16)  # 定义X坐标轴字体大小
plt.ylabel(u'Y', fontsize=16)  # 定义Y坐标轴字体大小
plt.title(u'距离曲线', fontsize=18)  # 定义标题文字和字体大小
plt.show()  # 画距离函数图像
x = Symbol('x')  # 约定变量
f = '+'.join(i for i in f)  # 定义函数
print(f)
result = solve(diff(f, x), x)  # 求解导数为0的点
coef = result[0].evalf()  # 取得result的数值
print(coef)  # 打印数值

x = np.linspace(-15, 15, num=500000)  # -15到15之间取500000个数据点，使图像连贯
y = coef * x  # 定义拟合的曲线的函数
plt.figure(figsize=(10, 5), facecolor='w')  # 定义图像大小和背景颜色白色
plt.plot(x, y, 'y.', markersize=1)
plt.plot(np.arange(3, 92), np.arange(4, 93), 'r.', markersize=8)
plt.xlabel(u'X', fontsize=16)  # 定义X坐标轴字体大小
plt.ylabel(u'Y', fontsize=16)  # 定义Y坐标轴字体大小
plt.title(u'拟合直线', fontsize=18)  # 定义标题文字和字体大小
plt.show()  # 画拟合直线图像
