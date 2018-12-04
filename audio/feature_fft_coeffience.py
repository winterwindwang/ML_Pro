import matplotlib.pyplot as plt
import numpy as np

def show(ori_func, ft, sampling_period=5):
    n = len(ori_func)
    interval = sampling_period / n   # 间隔
    # 绘制原始函数
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, sampling_period, interval), ori_func, 'black')
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    # 绘制变换后的函数
    plt.subplot(2, 1, 2)
    frequency = np.arange(n/2) / (n * interval)
    nfft = abs(ft[range(int(n / 2))] / n)
    plt.plot(frequency, nfft, 'red')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Amp. Spectrum')
    plt.show()

# 生成频率为 1 角速度为2 * pi 的正弦波
time = np.arange(0, 5, .006)
x = np.sin(2 * np.pi * 1 * time)
x2 = np.sin(2 * np.pi * 20 * time)
x3 = np.sin(2 * np.pi * 60 * time)
# 将其与频率20和60的波叠加起来
x += x2 + x3
y = np.fft.fft(x)
show(x, y)

# 生成方波，振幅是1，频率为10Hz
# 我们的间隔是0.05s， 每秒有200个点
# 所以需要每隔20个点设为1
x = np.zeros(len(time))
x[::20] = 1
y = np.fft.fft(x)
show(x, y)
