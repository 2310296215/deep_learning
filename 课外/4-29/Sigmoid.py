import tensorflow as tf

x = tf.linspace(-6., 6., 10)
print(x)
print(tf.nn.sigmoid(x))  # 通过sigmoid函数
# 可以看到，向量中元素值的范围由[−6,6]映射到(0,1)的区间
# Sigmoid 函数在输入值较大或较小时容易出现梯度值接
# 近于 0 的现象，称为梯度弥散现象
# 出现梯度弥散现象时，网络参数长时间得不到更新，
# 导致训练不收敛或停滞不动的现象发生，较深层次的网络模型中更容易出现梯度弥散现象。
