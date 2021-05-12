import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Himmelblau 函数是用来测试优化算法的常用样例函数之一
# 它包含了两个自变量𝑥和
# 𝑦，数学表达式是：𝑓(𝑥, 𝑦) = (𝑥2 + 𝑦 − 11)2 + (𝑥 + 𝑦2 − 7)2

def himmelblau(x):
    # himmelblau 函数实现，传入参数 x 为 2 个元素的List
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)  # 可视化的x坐标范围为-6~6
y = np.arange(-6, 6, 0.1)  # 可视化的y坐标范围为-6~6
print('x,y range:', x.shape, y.shape)
# 生成x-y平面采样网格点，方便可视化
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])  # 计算网格点上的函数值

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')  # 设置3D坐标轴
ax.plot_surface(X, Y, Z)  # 3D曲面图
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x = tf.constant([4., 0.])  # 初始化参数
# x = tf.constant([1., 0.])
# x = tf.constant([-4., 0.])
# x = tf.constant([-2., 2.])

for step in range(200):  # 循环优化200次
    with tf.GradientTape() as tape:  # 梯度跟踪
        tape.watch([x])  # 加入梯度跟踪列表
        y = himmelblau(x)  # 前向传播
    grads = tape.gradient(y, [x])[0]  # 反向传播
    x -= 0.01 * grads  # 更新参数， 0.01为学习率
    if step % 20 == 19:  # 打印优化的极小值
        print('step{}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))
