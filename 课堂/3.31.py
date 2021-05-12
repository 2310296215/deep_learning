import tensorflow as tf
tf.compat.v1.disable_eager_execution()
v1 = tf.Variable(tf.zeros((2, 2)), name='weights')
v2 = tf.Variable(tf.random.normal((2, 2)), name='random_weights')
init_top = tf.compat.v1.initialize_all_variables()
saver = tf.compat.v1.train.Saver()
def foo():
    with tf.compat.v1.Session() as sess:
        sess.run(init_top)
        print(sess.run(v1))
        print(sess.run(v2))
        save_path = saver.save(sess, 'D:/input/model.ckpt')
        print('model save in :', save_path)
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, 'D:/input/model.ckpt')
        print(sess.run(v1))
        print(sess.run(v2))
foo()
