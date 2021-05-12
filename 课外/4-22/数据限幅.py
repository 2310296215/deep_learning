import tensorflow as tf

x = tf.range(9)
print(tf.maximum(x, 2))  # 下限到2
print(tf.minimum(x, 7))  # 上限到7


def relu(x):  # Relu函数
    return tf.maximum(x, 0.)  # 下限为0即可


print(tf.minimum(tf.maximum(x, 2), 7))  # 限幅2-7

# 更方便地，我们可以使用 tf.clip_by_value 函数实现上下限幅：
print(tf.clip_by_value(x, 2, 7))
