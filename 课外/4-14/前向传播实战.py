import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as datasets

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    (x, y), (x_val, y_val) = datasets.mnist.load_data()  # 加载数据集
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # 转换为浮点张量，并放缩到-1~1
    y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转换为整型张量
    y = tf.one_hot(y, depth=10)  # 编码

    # 改变视图
    x = tf.reshape(x, [-1, 28 * 28])

    # 构建数据集对象
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(200)
    return train_dataset


def init_paramters():
    # 每一层的张量都需要被优化，所以使用Variable类型，并使用截断的正态分布初始化权值张量
    # 偏置向量初始化为0即可
    # 第一层参数
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))

    # 第二层参数
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))

    # 第三层参数
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    return w1, b1, w2, b2, w3, b3


def train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 第一层的运算 [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b,256] + [b, 256]
            h1 = x @ w1 + tf.broadcast_to(b1, (x.shape[0], 256))
            h1 = tf.nn.relu(h1)  # 通过激活函数

            # 第二层运算 [b, 256] => [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # 输出层计算 [b, 128] => [b, 10]
            out = h2 @ w3 + b3

            # 计算网络输出和标签之间的均方差  mse = mean(sum(y-out)^2)
            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)

            # 自动梯度， 需要求梯度的张量有[w1, b1, w2, b2, w3, b3]
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # 梯度更新，assign_sub 将当前值减去参数值，原地更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())

    return loss.numpy()


def train(epochs):
    losses = []
    train_dataset = load_data()
    w1, b1, w2, b2, w3, b3 = init_paramters()
    for epoch in range(epochs):
        loss = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001)
        losses.append(loss)

    x = [i for i in range(0, epochs)]
    # 绘制曲线
    plt.plot(x, losses, color='blue', marker='s', label='训练')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('MNIST数据集的前向传播训练误差曲线.png')
    plt.close()


if __name__ == '__main__':
    train(epochs=20)
