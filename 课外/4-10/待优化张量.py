import tensorflow as tf

a = tf.constant([-1, 0, 1, 2])
aa = tf.Variable(a)
print(aa.name, aa.trainable)  # name正常不需要用户去关注， trainable属性表示当前的张量是否需要优化，默认True,可以设置为False不优化

a_1 = tf.Variable([[1, 2], [3, 4]])
print(a_1)