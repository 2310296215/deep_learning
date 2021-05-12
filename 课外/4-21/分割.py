import tensorflow as tf
x = tf.random.normal([10, 38, 8])
# 等长分割成10份，在axis=0上
res = tf.split(x, num_or_size_splits=10, axis=0)
print(len(res))

# 自定义长度分割
res = tf.split(x, num_or_size_splits=[4, 2, 2, 2], axis=0)
print(len(res))
print(res[0])

# 在某个维度上全部按照长度为1来分割
res = tf.unstack(x, axis=1)
print(len(res))
