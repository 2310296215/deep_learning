import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def foo():
    with tf.name_scope('Scope_A'):
        a = tf.add(1, 2, name='A_add')
        b = tf.multiply(a, 3, name='A_mul')
    with tf.name_scope('Scope_B'):
        c = tf.add(4, 5, name='B_add')
        d = tf.multiply(c, 6, name='B_mul')
    e = tf.add(b, d, name='output')
    writer = tf.compat.v1.summary.FileWriter('./name_scope', graph=tf.compat.v1.get_default_graph())
    writer.close()


foo()
