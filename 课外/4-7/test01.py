import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiti']
matplotlib.rcParams['axes.unicode_minus'] = False

import tensorflow as tf
import timeit

cpu_data = []
gpu_data = []
for n in range(9):
    # 创建在cpu上运行的两个矩阵
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([1, n])
        cpu_b = tf.random.normal([n, 1])
        print(cpu_a.device, cpu_b.device)

    # 创建在gpu上运行的两个矩阵
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([1, n])
        gpu_b = tf.random.normal([n, 1])
        print(gpu_a.device, gpu_b.device)


    def cpu_run():
        with tf.device('/cpu:0'):
            c = tf.matmul(cpu_a, cpu_b)
        return c


    def gpu_run():
        with tf.device('/gpu:0'):
            c = tf.matmul(gpu_a, gpu_b)
        return c


    # 第一次计算需要热身，避免将初始化时间计算在内
    cpu_time = timeit.timeit(cpu_run, number=10)
    gpu_time = timeit.timeit(gpu_run, number=10)
    print('warmup:', cpu_time, gpu_time)
    # 正式计算10次取平均时间
    cpu_time = timeit.timeit(cpu_run, number=10)
    gpu_time = timeit.timeit(gpu_run, number=10)
    print('run time:', cpu_time, gpu_time)
    cpu_data.append(cpu_time / 10)
    gpu_data.append(gpu_time / 10)

    del cpu_a, cpu_b, gpu_b, gpu_a

x = [10 ** i for i in range(9)]
cpu_data = [1000 * i for i in cpu_data]
gpu_data = [1000 * i for i in gpu_data]
plt.plot(x, cpu_data, 'C1')
plt.plot(x, cpu_data, color='C1', marker='s', label='CPU')
plt.plot(x, gpu_data, 'C0')
plt.plot(x, gpu_data, color='C0', marker='^', label='GPU')

plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
# plt.ylim([0, 100])
plt.xlabel('矩阵大小n:(1xn)@(nx1)')
plt.ylabel('运算时间(ms)')
plt.legend()
plt.savefig('gpu-time.svg')
