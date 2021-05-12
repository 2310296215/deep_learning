import tensorflow as tf

a = tf.random.normal([4, 35, 8])
b = tf.random.normal([6, 35, 8])
print(tf.concat([a, b], axis=0))  # 在axis=0的维度上拼接

a = tf.random.normal([4, 35, 8])
b = tf.random.normal([4, 35, 2])
print(tf.concat([a, b], axis=2))  # 在axis=2的维度上拼接

# 拼接操作直接在现有维度上合并数据，并不会创建新的维度。如果在合并数据
# 时，希望创建一个新的维度，则需要使用 tf.stack 操作。

a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
print(tf.stack([a, b], axis=0))
print(tf.stack([a, b], axis=-1))
