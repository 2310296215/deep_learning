import tensorflow as tf

x = tf.random.normal(4, 784)
# 隐藏层1张量
w1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

# 隐藏层2张量
w2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

# 隐藏层3张量
w3 = tf.Variable(tf.random.normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))

# 输出层张量
w4 = tf.Variable(tf.random.normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

with tf.GradientTape() as tape:

    # 隐藏层1
    h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
    h1 = tf.nn.relu(h1)

    # 隐藏层2
    h2 = h1 @ w2 + b2
    h2 = tf.nn.relu(h2)

    # 隐藏层3
    h3 = h2 @ w3 + b3
    h3 = tf.nn.relu(h3)

    # 输出层
    h4 = h3 @ w4 + b4
    # 最后一层是否需要添加激活函数通常视具体的任务而定，这里加不加都可以


