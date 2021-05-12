import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import os
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)


# 预处理函数
def preprocess(x, y):
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.  # 归一化
    x = tf.reshape(x, [-1, 28 * 28])  # 平整化数据
    y = tf.cast(y, dtype=tf.int32)  # 转换类型
    y = tf.one_hot(y, depth=10)  # one-hot编码

    return x, y


(x, y), (x_test, y_test) = datasets.mnist.load_data()

# 训练数据
batchsz = 512
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000)  # 打乱数据集
train_db = train_db.batch(batchsz)  # 512批样本并行计算
train_db = train_db.map(preprocess)  # 对数据进行预处理
train_db = train_db.repeat(20)  # 每训练20次，算一次迭代

# 测试数据
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
x, y = next(iter(train_db))
print('train sample；', x.shape, y.shape)


# 运行主函数
def main():
    # 学习率
    lr = 1e-2
    accs, losses = [], []

    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    for step, (x, y) in enumerate(train_db):

        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape:

            # 第一层
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # 第二层
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # 输出层
            out = h2 @ w3 + b3

            # 误差的平方
            loss = tf.square(y - out)
            # 平均误差
            loss = tf.reduce_mean(loss)

        # 自动梯度训练
        grades = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # 梯度更新参数
        for p, g in zip([w1, b1, w2, b2, w3, b3], grades):
            p.assign_sub(lr * g)

        if step % 80 == 0:  # 每80step打印误差
            print(step, 'loss:', float(loss))
            losses.append(float(loss))

        if step % 80 == 0:  # 每80次step后检测一次
            total, total_correct = 0., 0

            for x, y in test_db:
                # 第一层
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # 第二层
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # 输出层
                out = h2 @ w3 + b3

                # 获得预测概率最大的类别与真实值是否相同
                pred = tf.argmax(out, axis=1)
                y = tf.argmax(y, axis=1)
                correct = tf.equal(pred, y)
                # 布尔 转 int 转 numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()  # 求预测正确的一共有多少
                total += x.shape[0]  # 总数

            print(step, 'Evalute Acc:', total_correct / total)
            accs.append(total_correct / total)

    plt.figure()
    x = [i * 80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('train.svg')

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='测试')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('test.svg')


if __name__ == '__main__':
    main()
