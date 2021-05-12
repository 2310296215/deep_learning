import tensorflow as tf

tf.compat.v1.disable_eager_execution()
state = tf.Variable(1, name='counter')
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.initialize_all_variables())
    for i in range(1, 101):
        new_value1 = tf.multiply(state, tf.constant(i))
        update = tf.compat.v1.assign(state, new_value1)
        sess.run(update)
        print(sess.run(state))
