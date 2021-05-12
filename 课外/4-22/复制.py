import tensorflow as tf

x = tf.random.normal([3, 32, 32, 3])
print(tf.tile(x, multiples=[2, 3, 3, 1]))  # 图片的宽高复制两份，图片数量复制一份，通道不复制
