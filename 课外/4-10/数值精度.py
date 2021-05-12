import tensorflow as tf

a = tf.constant(123456789, dtype=tf.int16)
b = tf.constant(123456789, dtype=tf.int32)
print(a)
print(b)
print(a._id)  # id表示tensorflow中对象索引编号
print(b._id)

import numpy as np

pi = np.pi
x = tf.constant(pi, dtype=tf.float32)
y = tf.constant(pi, dtype=tf.float64)  # 不同的精度，显示的长度不一样
print(x)
print(y)


print('before:', a.dtype)
if a.dtype != tf.float64:
    a = tf.cast(a, tf.float64)  # 精度转换，也可以类型转换-
print('after:', a.dtype)

# 当高精度的张量转换为低精度的张量，数值会溢出
x_1 = tf.constant(123456789, dtype=tf.int64)
y_1 = tf.cast(x_1, dtype=tf.int16)
print(x_1)
print(y_1)

# 布尔类型于整型之间相互转换也是合法的

a_1 = tf.constant([True, False])
print(tf.cast(a_1, dtype=tf.int32))