import tensorflow as tf

a = tf.constant(True)  # 创建布尔型标量
print(a)

a_1 = tf.constant([True, False])  # 创建布尔型向量
print(a_1)

test = True
print(a == test)
print(a is test)  # tf中布尔类型仅数值比较相同，对象不等价


