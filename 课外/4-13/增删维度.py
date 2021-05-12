import tensorflow as tf

x = tf.random.uniform([28, 28], maxval=10, dtype=tf.int32)

# 增加维度
x = tf.expand_dims(x, axis=2)  # axis=2表示宽维度后面的维度，插入一个维度
print(x)
x = tf.expand_dims(x, axis=0)  # 在高纬度之前插入新的维度，axis为正的时候表示维度之前，当为负数，表示之后，用法和索引差不多
print(x)

# 删除维度
x = tf.squeeze(x, axis=0)  # 如果不指定参数axis，则默认删除所有长度为1的维度
print(x)
x = tf.squeeze(x, axis=2)
print(x)