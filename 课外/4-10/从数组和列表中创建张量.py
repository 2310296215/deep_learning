import tensorflow as tf
import numpy as np

x = tf.convert_to_tensor([1., 3.])
print(x)
y = tf.convert_to_tensor(np.array([[1, 2], [3, 4]]))
print(y)

print(tf.ones([2, 2]))
print(tf.zeros([3, 3]))

print(tf.zeros_like(y))  # 创建和y形状相同,值为0的张量

# 创建自定义数值张量
print(tf.fill([2, 3], 2))

# 创建标准正态分布的张量， shape表示形状，mean表示均值， stddev表示标准差
print(tf.random.normal([2, 2], mean=0, stddev=1))

# 创建在一定区间内均匀分布的张量
print(tf.random.uniform([2, 3], maxval=10, dtype=tf.int32))  # 创建采样自[0, 10)均匀分布的整型数据的矩阵，默认为[0, 1)

print(tf.random.normal([2, 4]))
