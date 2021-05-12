import tensorflow as tf

# ReLU 函数在𝑥 < 0时导数值恒为 0，也可能会造成梯度弥散现象，为了克服这个问题，LeakyReLU 函数被提出
x = tf.linspace(-6., 6., 10)
print(tf.nn.leaky_relu(x, alpha=0.1))  # 当alpha=0时就是ReLU
