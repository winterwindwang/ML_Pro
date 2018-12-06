#coding=utf-8
#测试文件
from sigprocess import *
from calcmfcc import *
import librosa
import matplotlib.pyplot as plt
import numpy as np
import audioread

# example
# t = np.arange(256)
# sp = np.fft.fft(np.sin(t))
# freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real, freq, sp.imag)


def second_order_derivative(signal):
    return np.array([signal[i+1] - 2*signal[i] - signal[i-1] for i in range(len(signal)-2) if i>=1])

input_file_cover = r'D:\student\隐写分析\数据\1second_cover_data\1_1.wav'
input_file_stego = r'D:\student\隐写分析\数据\1second_stego_data\1_1_stego.wav'
cover_signal, cover_rate = librosa.load(input_file_cover,sr=None)
stego_signal, cover_rate = librosa.load(input_file_stego,sr=None)

# comupte second order derivated
cover_signal = second_order_derivative(cover_signal)
stego_signal = second_order_derivative(stego_signal)

cover_signal_fft = np.fft.fft(cover_signal)
stego_signal_fft = np.fft.fft(stego_signal)
# 取实部
cover_signal_fft_real = abs(cover_signal_fft.real)
stego_signal_fft_real = abs(stego_signal_fft.real)
# 差异
diff_signal_fft = stego_signal_fft_real-cover_signal_fft_real

plt.figure()
plt.subplot(3,1,1)
plt.plot(np.arange(len(cover_signal)), cover_signal_fft_real*1000000)
plt.title('spectrum cover')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')
# plt.show()

plt.subplot(3,1,2)
plt.plot(np.arange(len(stego_signal)), stego_signal_fft_real*1000000)
plt.title('spectrum stego')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')

plt.subplot(3,1,3)
plt.plot(np.arange(len(stego_signal)), diff_signal_fft*1000000)
plt.title('spectrum differ')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')

plt.show()



# 画出频谱



