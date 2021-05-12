import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x_test:', x_test.shape, 'y_test:', y_test.shape)

# 数据加载进入内存后，需要转换成 Dataset 对象，才能利用 TensorFlow 提供的各种便捷功能
train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 构建Dataset对象

# 通过 Dataset.shuffle(buffer_size)工具可以设置 Dataset 对象随机打散数据之间的顺序，
# 防止每次训练时数据按固定顺序产生，从而使得模型尝试“记忆”住标签信息
# buffer_size 参数指定缓冲池的大小，一般设置为一个较大的常数即可
train_db = train_db.shuffle(10000)  # 随机打乱样本，不会打乱样本与标签映射关系

# 为了利用显卡的并行计算能力，一般在网络的计算过程中会同时计算多个样本，我们
# 把这种训练方式叫做批训练，其中一个批中样本的数量叫做 Batch Size。为了一次能够从
# Dataset 中产生 Batch Size 数量的样本，需要设置 Dataset 为批训练方式
train_db = train_db.batch(128)  # 设置批训练，batch size为128，一次并行计算 128 个样本的数据

# 自定义预处理函数
def preorocess(x, y):
    # 调用此函数时会自动传入 x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1., 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    # 返回的 x,y 将替换传入的 x,y 参数，从而实现数据的预处理功能
    return x, y

# 循环训练

