import tensorflow as tf
from tensorflow import keras

# 通过 tf.reduce_max、tf.reduce_min、tf.reduce_mean、tf.reduce_sum 函数可以求解张量
# 在某个维度上的最大、最小、均值、和，也可以求全局最大、最小、均值、和信息。
x = tf.random.normal([4, 8])
print(tf.reduce_max(x, axis=1))  # 不指定axis参数，就会面向全局
print(tf.reduce_min(x, axis=1))
print(tf.reduce_mean(x, axis=1))
print(tf.reduce_sum(x, axis=-1))  # 求最有一个维度的和

# mse求误差
out = tf.random.normal([4, 10])  # 模拟网络输出
y = tf.constant([1, 2, 2, 0])  # 模拟真实标签
y = tf.one_hot(y, depth=10)  # 编码
loss = keras.losses.mse(y, out)  # 计算每个样本的误差
loss = tf.reduce_mean(loss)  # 计算平均误差
print('平均误差：', loss)

# 除了希望获取张量的最值信息，还希望获得最值所在的位置索引号
out = tf.random.normal([2, 10])
print(tf.nn.softmax(out, axis=1))  # 通过 softmax 函数转换为概率值

# 通过 tf.argmax(x, axis)和 tf.argmin(x, axis)可以求解在 axis 轴上，x 的最大值、最小值所
# 在的索引号
out = tf.random.normal([2, 10])
out = tf.nn.softmax(out, axis=1)
print(out)
print(tf.argmax(out, axis=1))  # 求每个样本最大概率值的索引
