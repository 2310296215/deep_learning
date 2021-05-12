import tensorflow as tf

aa = tf.constant(1.2)
print(aa)

x_1 = tf.constant([1.2])
print(x_1)
print(x_1.shape)

x_2 = tf.constant([1.2, 2.3, 3.4])
print(x_2)
print(x_2.numpy())
print(x_2.shape)

x_3 = tf.constant([[[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]])
print(x_3)
print(x_3.numpy())
print(x_3.shape)