import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal([4, 28 * 28])
# 创建全连接层，指定输出节点数和激活函数
fc = layers.Dense(256, activation=tf.nn.relu)
h1 = fc(x)
print(h1)
print('权值矩阵：', fc.kernel)
print('偏置向量：', fc.bias)
print('返回待优化参数列表：', fc.trainable_variables)
print('所有内部张量列表：', fc.variables)
