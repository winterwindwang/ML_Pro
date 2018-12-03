import matplotlib.pyplot as plt
import numpy as np

# loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
#
# fname要读取的文件、文件名、或生成器。
# dtype数据类型，默认float。
# comments注释。
# delimiter分隔符，默认是空格。
# skiprows跳过前几行读取，默认是0，必须是int整型。
# usecols：要读取哪些列，0是第一列。例如，usecols = （1,4,5）将提取第2，第5和第6列。默认读取所有列。
# unpack如果为True，将分列读取。

mp3stego = np.loadtxt(r'D:\student\隐写分析\MP3Stego_1_1_18\实验尝试\不含密载体_频谱图.txt',dtype=np.float32,skiprows=1)
mp3stego_hided = np.loadtxt(r'D:\student\隐写分析\MP3Stego_1_1_18\实验尝试\含密载体_频谱图.txt',dtype=np.float32,skiprows=1)

# mp3stego = np.array([[1,2],[2,3],[3,4]])
# mp3stego_hided = np.array([[1,3],[2,4],[3,5]])

mp3stego_hided = mp3stego_hided-mp3stego

print(mp3stego_hided)

# 查看矩阵的大小
# print(mp3stego.shape)

# mp3stego_set = set([tuple(t) for t in mp3stego])
# mp3stego_hided_set = set([tuple(t) for t in mp3stego_hided])
# mp3stego_hided_differ = np.array([list(t) for t in (mp3stego_hided_set-mp3stego_set)])

# print('信息隐藏前后的差距\n',(mp3stego_hided_set-mp3stego_set))