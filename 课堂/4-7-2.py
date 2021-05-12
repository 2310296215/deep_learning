import tensorflow as tf
tf.compat.v1.disable_eager_execution()

juzhen = tf.Variable(tf.random.normal((3, 3), stddev=0.35))
ones = tf.ones((3, 3))
rand = tf.Variable(tf.random.normal(()))
xuehao = tf.constant(41, dtype=tf.float32)
add_1 = tf.add(xuehao, rand)
mul = tf.matmul(juzhen, ones)
add_2 = tf.add(add_1, mul)
save = tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(mul))
    print(sess.run(add_1))
    print(sess.run(add_2))
    save.save(sess, 'C:/result/model.ckpt')
    writer = tf.compat.v1.summary.FileWriter('C:/graph/name_scope', graph=tf.compat.v1.get_default_graph())
    writer.close()

