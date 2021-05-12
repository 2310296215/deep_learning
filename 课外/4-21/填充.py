import tensorflow as tf
from tensorflow import keras

a = tf.constant([1, 2, 3, 4, 5, 6])  # 第一个句子
b = tf.constant([2, 3, 4, 5])  # 第二个句子
b = tf.pad(b, [[0, 2]])  # 在第一个的句子末尾填充
print(a, b)

out = tf.stack([a, b], axis=0)
print(out)
print('-' * 20)

total_words = 10000  # 设定词汇量大小
max_review_len = 80  # 最大句子长度
embedding_len = 100  # 词向量长度
# 加载数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)

# 将句子填充或截到相同长度，设置为末尾填充和末尾截断方式
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len, truncating='post', padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len, truncating='post', padding='post')
print(x_train.shape, x_test.shape)
print(x_test[0])
print('-' * 20)

x = tf.random.normal([4, 28, 28, 1])
print(tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]]))