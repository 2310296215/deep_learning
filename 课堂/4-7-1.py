import tensorflow as tf

tf.compat.v1.disable_eager_execution()
state = tf.Variable(0, name='add', dtype=tf.int64)
num = tf.Variable(1, name='mul', dtype=tf.int64)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.initialize_all_variables())
    for i in range(1, 11):
        t = tf.constant(i, dtype=tf.int64)
        num = num * t
        state = state + num
        print(sess.run(state))
