import tensorflow as tf

# a = tf.ones([3, 3])
# b = tf.zeros([3, 3])
# cond = tf.constant([[True, False, False], [False, True, False], [False, False, True]])
# print(tf.where(cond, a, b))
# print(tf.eye(3))

# 获取正数
# x = tf.random.normal([3, 3])
# mask = x > 0
# indices = tf.where(mask)
# print(tf.gather_nd(x, indices))
# print(tf.boolean_mask(x, mask))

# scatter_nd
# indices = tf.constant([[4], [3], [1], [7]])
# updates = tf.constant([4.4, 3.3, 1.1, 7.7])
# print(tf.scatter_nd(indices, updates, [8]))

# indices = tf.constant([[1], [3]])
# updates = tf.constant([  # 构造写入数据，即 2 个矩阵
#     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
#     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
# ])
# print(tf.scatter_nd(indices, updates, [4, 4, 4]))

# meshgrid
x = tf.linspace(-8., 8, 100)
y = tf.linspace(-8., 8, 100)
x, y = tf.meshgrid(x, y)
# print(x.shape, y.shape)
z = tf.sqrt(x ** 2 + y ** 2)
z = tf.sin(z) / z
import matplotlib
from matplotlib import pyplot as plt
# 导入 3D 坐标轴支持
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()
