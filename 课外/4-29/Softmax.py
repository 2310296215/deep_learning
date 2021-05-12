import tensorflow as tf

# 输出值𝑜𝑖 ∈ [0,1]，且所有输出值之和为 1，这种设定以多分类问题最为常见
x = tf.constant([2., 1., 0.1])
print(tf.nn.softmax(x))

# 在 Softmax 函数的数值计算过程中，容易因输入值偏大发生数值溢出现象；在计算交叉熵时，
# 也会出现数值溢出的问题。为了数值计算的稳定性，TensorFlow 中提供了一个统一的接口，
# 将 Softmax 与交叉熵损失函数同时实现，同时也处理了数值不稳定的异常，
# 一般推荐使用这些接口函数，避免分开使用 Softmax 函数与交叉熵损失函数。
# 函数式接口为tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
z = tf.random.normal([2, 10])  # 构造输出层的输出
y_onehot = tf.constant([1, 3])  # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10)

loss = tf.keras.losses.categorical_crossentropy(y_onehot, z, from_logits=True)
loss = tf.reduce_mean(loss)  # 计算平均交叉熵损失
print(loss)

criteon = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot, z)
print(loss)