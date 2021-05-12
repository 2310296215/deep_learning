import tensorflow as tf

out = tf.random.normal([100, 10])
out = tf.nn.softmax(out, axis=1)  # 输出转换为概率
pred = tf.argmax(out, axis=1)  # 计算预测值
print('预测值：', pred)
y = tf.random.uniform([100], dtype=tf.int64, maxval=10)
print('真实值：', y)

# 通过 tf.equal(a, b)(或 tf.math.equal(a, b)，两者等价)函数可以比较这 2 个张量是否相等
out = tf.equal(pred, y)  # 预测值和真实值的比较，返回布尔类型的张量
print(out)

out = tf.cast(out, dtype=tf.float32)  # 布尔类型转int
correct = tf.reduce_sum(out)  # 计算有多少True
print(correct)