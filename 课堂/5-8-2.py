import numpy as np  # 引入致值计算的包
from mpl_toolkits.mplot3d import Axes3D  # 引入丽3D三维图像的包
import matplotlib as mpl
import matplotlib.pyplot as plt  # 引入画图的包
from sympy import *  # 引入求导致的包

mpl.rcParams['font.sans-serif'] = [u'simHei']  # 定义字体微软雅黑
mpl.rcParams['axes.unicode_minus'] = False  # 使得函数图像坐标轴上的负坐标的负号能正常显示
x = np.arange(-5, 5, 0.25)  # 定义X坐标轴范围
y = np.arange(-5, 5, 0.25)  # 定义Y坐标轴范围
x, y = np.meshgrid(x, y)  # 定义网格
z = 0
for i in range(3, 92):
    z += (i * x + y - (i + 1)) ** 2
ax = plt.figure(figsize=(10, 5), facecolor='w').gca(projection='3d')  # 定义三维图像大小和背景颜色白色
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('gray_r'), linewidth=0, antialiased=False)  # 定义图像外观
a = 5  # 定义梯度下降起始点的X坐标
b = 5  # 定义梯度下降起始点的Y坐标
c = 0
for i in range(3, 92):
    c += (i * x + y - (i + 1)) ** 2
ax.scatter(a, b, c, c='green')  # 用绿色标注起始点
learning_rate = 0.00000008  # 定义学习率
d = a  # 将a的数值存储起来
e = b  # 将b的数值存储起来
a = Symbol('a')  # 符号化a
b = Symbol('b')  # 符号化b
d1 = Symbol('d')  # 符号化a
e1 = Symbol('e')  # 符号化b
f1 = Symbol('f')  # 符号化a
g1 = Symbol('g')  # 符号化b
s = m = n = 0
for i in range(3, 92):
    s += (i * a + g1 - (i + 1)) ** 2
    m += (i * f1 + b - (i + 1)) ** 2
    n += (i * d1 + e1 - (i + 1)) ** 2
for _ in range(5000):  # 迭代循环次数
    print(_)
    f = d  # 将d的致值存储起来
    g = e  # 将e的数值存储起来
    d = f - learning_rate * (diff(s, a).subs([('a', f), ('g', g)]))  # 以一定的学习率梯度下降，得到下降后的X坐标
    e = g - learning_rate * (diff(m, b).subs([('b', g), ('f', f)]))  # 以一定的学习率梯度下降，得到下降后的Y坐标
    c = n.subs([('d', d), ('e', e)])  # 梯度下降后的Z坐标
    if _ == 4999:  # 定义最后一次选代循环后的标记点和输出的内容
        ax.scatter(d, e, c, c='red')  # 用红色标记最后—次迭代循环后得到的点
        print(d, e)  # 打印最后一次运代循环后点的X,Y坐标
        print(f, g)  # 打印倒数第二次迭代循环后点的X,Y坐标
        print(1.25, 1.73)  # 打印真实的距离最小值点的X,Y坐标
    else:  # 定义中间每次迭代循环后的标记点
        ax.scatter(d, e, c, c='green')  # 用绿色标记中间每次迭代循环后得到的点
plt.xlabel(u'X', fontsize=16)  # 定义X坐标轴字体大小
plt.ylabel(u'Y', fontsize=16)  # 定义Y坐标轴字体大小
plt.title(u'梯度下降法找最小值点', fontsize=18)  # 定义标题文字和字体大小
plt.show()  # 画距离函致图像并标记所有梯度下降的点
x = np.linspace(0, 100, num=500080)  # 0到10之间取500000个点，.使图像连贯
y = d * x + e  # 定义拟合的曲线的函致
plt.figure(figsize=(10, 5), facecolor='w')  # 定义图像大小和背景颜色白色
plt.plot(x, y, 'y.', markersize=1)  # 用黄色画这500888个点
plt.plot(np.arange(3, 92), np.arange(4, 93), 'r.', markersize=8)
plt.xlabel(u'x', fontsize=16)  # 定义X坐标轴字体大小
plt.ylabel(u'Y', fontsize=16)  # 定义Y坐标轴字体大小
plt.title(u"拟合曲线", fontsize=18)  # 定义标题文字和字体大小
plt.show()  # 画拟合直线图像
