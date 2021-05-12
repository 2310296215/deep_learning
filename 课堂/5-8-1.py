import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

x_data = np.array([2, 4, 6]).astype(np.float32)
y_data = np.array([3, 5, 8]).astype(np.float32)

# x_data = np.random.rand(100).astype(np.float32)
# y_data = 0.1 * x_data + 0.3

weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optim = tf.compat.v1.train.GradientDescentOptimizer(0.05)
train = optim.minimize(loss)

init = tf.compat.v1.initialize_all_variables()
sess = tf.compat.v1.Session()
sess.run(init)

for step in range(5001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weights), sess.run(biases))
