# 在搭建完模型结构后，下一步就是选择合适的误差函数来计算误差。常见的误差函数
# 有均方差、交叉熵、KL 散度、Hinge Loss 函数等，其中均方差函数和交叉熵函数在深度学
# 习中比较常见，均方差函数主要用于回归问题，交叉熵函数主要用于分类问题。
import tensorflow as tf

o = tf.random.normal([2, 10])  # 输出值
y_onehot = tf.constant([2, 3])  # 真实值
y_onehot = tf.one_hot(y_onehot, depth=10)
loss = tf.keras.losses.mse(y_onehot, o)  # 计算均方差
print(loss)

# 特别要注意的是，MSE 函数返回的是每个样本的均方差，需要在样本维度上再次平均来获
# 得平均样本的均方差
loss = tf.reduce_mean(loss)  # 计算batch均方差
print(loss)

# 也可以通过层方式实现，对应的类为 keras.losses.MeanSquaredError()，和其他层的类一
# 样，调用__call__函数即可完成前向计算，
criton = tf.keras.losses.MeanSquaredError()
loss = criton(y_onehot, o)
print(loss)