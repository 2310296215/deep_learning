import tensorflow as tf

x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 4个班级，每个35人，8门学课
# print(tf.gather(x, [0, 1], axis=0))  # 取第1-2个班级的名单
#
# # 抽查所有班级的第 1、4、9、12、13、27 号同学的成绩数据
# print(tf.gather(x, [0, 3, 8, 11, 12, 26], axis=1))
#
# # 收集第3门和第5门学课成绩
# print(tf.gather(x, [2, 4], axis=2))
# a = tf.range(8)
# a = tf.reshape(a, [4, 2])
# print(a)
# print(tf.gather(a, [3, 1, 2, 0], axis=0))  # 收集4，2，3，1号数据

# class_ = tf.gather(x, [1, 2], axis=0)  # 先抽取2，3班级
# grade = tf.gather(class_, [2, 3, 5, 26], axis=1)  # 再抽取对应学号的学生
# print(grade)

# 抽查第二个班级的第二个学生，第三个班级的第三个学生，第四个班级的第四个学生
# print(tf.stack([x[1, 1], x[2, 2], x[3, 3]], axis=0))
#
# print(tf.gather_nd(x, [[1, 1], [2, 2], [3, 3]]))
# print(tf.gather_nd(x, [[1, 1, 2], [2, 2, 4], [3, 3, 2]]))

# print(tf.boolean_mask(x, mask=[True, False, False, True], axis=0))  # 取1，4班级。长度必须和对应的维度的长度一样
y = tf.random.normal([2, 3, 8])
print(tf.boolean_mask(y, [[True, True, False], [False, True, True]]))
print(tf.gather_nd(y, [[0, 0], [0, 1], [1, 1], [1, 2]]))
