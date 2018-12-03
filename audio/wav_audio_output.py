import wave
import os
import numpy as np
import matplotlib.pyplot as plt
import struct

#wave 文件读入

file_path = r"D:\student\隐写分析\音频"
input_path = file_path+r'\p225_025.wav'
fstr = wave.open(input_path,'rb')
params = fstr.getparams()
nchannels,samwidth,framerate,nframes = params[:4]
audioStr = fstr.readframes(nframes)
audioData = np.fromstring(audioStr,dtype=np.int16)
audioData = audioData *1.0 / sum(abs(audioData))
audioData = np.reshape(audioData,[nframes,nchannels]).T
fstr.close()

# plot the wave
plt.specgram(audioData[0],Fs=nframes,scale_by_freq=True,sides='default')
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.show()


# 文件写入

# outData = audioData
# out_path = file_path + r'\output.wav'
# outwave = wave.open(out_path,'wb')
# nchannels = 1
# sampwidth = 2
# framerates = 8000
# dataSize = len(outData)
# framerate = int(framerates)
# nframes = dataSize
# comptype = 'NONE'
# compname = 'not compress'
# outwave.setparams((nchannels,sampwidth,framerate,nframes,comptype,compname))
#
# for v in outData:
#     outwave.writeframes(struct.pack('h',int(v * 64000 / 2)))    #outData:16位，-32767~32767，注意不要溢出
# outwave.close()

