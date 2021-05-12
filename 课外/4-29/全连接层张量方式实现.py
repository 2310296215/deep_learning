import tensorflow as tf

x = tf.random.normal([2, 784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x, w1) + b1  # 线性变换
o1 = tf.nn.relu(o1)  # 激活函数
print(o1)
