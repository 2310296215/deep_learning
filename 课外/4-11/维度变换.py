import tensorflow as tf

x = tf.range(96)
print(x)
x = tf.reshape(x, [2, 4, 12])
print(x)