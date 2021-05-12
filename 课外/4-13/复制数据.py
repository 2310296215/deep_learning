import tensorflow as tf

# 需要注意的是，tf.tile()是创建一个新的张量来保存复制后的张量，计算代价较高
# 可以通过 tf.tile(x, multiples)函数完成数据在指定维度上的复制操作，multiples 分别指
# 定了每个维度上面的复制倍数，对应位置为 1 表明不复制，为 2 表明新长度为原来长度的
# 2 倍，即数据复制一份，以此类推。
b = tf.constant([1, 2])  # 创建向量
print(b)
b = tf.expand_dims(b, axis=0)  # 插入新维度变成矩阵
print(b)

# 复制数据
b = tf.tile(b, multiples=[2, 1])  # 在axis=0的维度下复制1次，在axis=0的维度下不变
print(b)

x = tf.range(4)
x = tf.reshape(x, [2, 2])
print(x)
x = tf.tile(x, multiples=[2, 1])
print(x)
x = tf.tile(x, multiples=[1, 2])
print(x)

