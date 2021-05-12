import tensorflow as tf

# Tanh 函数能够将𝑥 ∈ 𝑅的输入“压缩”到(−1,1)区间
x = tf.linspace(-6., 6., 10)
print(tf.nn.tanh(x))
