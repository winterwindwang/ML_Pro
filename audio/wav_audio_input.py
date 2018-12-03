import numpy as np
import os
import wave
import matplotlib.pyplot as plt

file_path = r'D:\student\隐写分析\音频\human\p225'  # 添加路径
file_path_1 = r'D:\student\隐写分析\音频\p225_025.wav'

# file_names = os.listdir(file_path_1)
# print(len(file_names))

wav_audio = wave.open(file_path_1,'rb')      # 'rb'  'wb'
paramas = wav_audio.getparams()

# nchannels:声道数
# sampwidth:量化位数（byte）
# framerate:采样频率  16k
# nframes:采样点数
nchannels,sampwidth,framerate,nframes = paramas[:4]
strData = wav_audio.readframes(nframes)     # 读取音频，字符串格式
waveData = np.fromstring(strData,dtype=np.int16)  # 将字符串转化为int
waveData = waveData*1.0/sum(abs(waveData))      # 将幅值归一化

print(nchannels)

# plot the wave
# print(framerate)
# time = np.arange(0,nframes)*(1.0 / framerate)
# plt.figure()
# plt.plot(time,waveData)
# plt.xlabel('Time(s)')
# plt.ylabel('Amplitude')
# plt.title('single channel wavedata')
# plt.grid('on')  # 标尺: on：有  off：无
# plt.show()
audio_path_1 = r'D:\student\隐写分析\音频\123.wav'

multiply_audio = wave.open(audio_path_1,'rb')
paramas_1 = multiply_audio.getparams()
nchannels,sampwidth,framerate,nframes = paramas_1[:4]
# 读取音频字符串
str_multiply_audio_data = multiply_audio.readframes()
# 将音频字符串转化为整型
multiply_audio_data = np.fromstring(str_multiply_audio_data,dtypes=np.int16)
# 将幅值归一化
multiply_audio_data = multiply_audio_data * 1.0 / sum(abs(multiply_audio_data))

multiply_audio_data = np.reshape(multiply_audio_data,[nframes,nchannels])

multiply_audio.close()

# plot the wave
time = np.arange(0,nframes) * (1/framerate)
plt.figure()

plt.subplot(5,1,1)
plt.plot(time,multiply_audio_data[:,0])
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('ch1 wavedata')

plt.subplot(5,1,3)
plt.plot(time,multiply_audio_data[0:1])
plt.xlabel('Time(s)')
plt.ylabel('Ampltiude')
plt.title('ch2 wavedata')

plt.subplot(5,1,5)
plt.plot(time,multiply_audio_data[0:2])
plt.xlabel('Time(s)')
plt.ylabel('Ampltiude')
plt.title('ch3 wavedata')

plt.gri('on')  # 标尺：on：有，off：没有

plt.show()