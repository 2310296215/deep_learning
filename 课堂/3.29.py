import tensorflow as tf
tf.compat.v1.disable_eager_execution()
data = tf.eye(3, 3, dtype=tf.int32)
with tf.compat.v1.Session() as sess:
    print(sess.run(data))
