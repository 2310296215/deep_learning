import tensorflow as tf

tf.compat.v1.disable_eager_execution()
in1 = tf.constant(3.0)
in2 = tf.constant(2.0)
in3 = tf.constant(5.0)
temp = tf.add(in1, in2)
mul = tf.multiply(temp, in3)


def foo():
    with tf.compat.v1.Session() as sess:
        result = sess.run([mul, temp])
        print(result)


foo()
