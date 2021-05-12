import tensorflow as tf
import numpy as np

x = tf.ones([2, 2])
print(tf.norm(x, ord=1))  # 计算L1范数
print(tf.norm(x, ord=2))  # 计算L2范数
print(tf.norm(x, ord=np.inf))  # 计算oo范数
