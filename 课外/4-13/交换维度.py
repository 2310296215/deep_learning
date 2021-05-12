import tensorflow as tf

# 交换维度
y = tf.random.normal([2, 32, 32, 3])
print(y)
y = tf.transpose(y, perm=[0, 3, 1, 2])
print(y)

# 需要注意的是，通过 tf.transpose 完成维度交换后，张量的存储顺序已经改变，视图也
# 随之改变，后续的所有操作必须基于新的存续顺序和视图进行。相对于改变视图操作，维
# 度交换操作的计算代价更高。