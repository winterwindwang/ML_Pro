import wave
import os
import numpy as np
import matplotlib.pyplot as plt

CutTimeOf = 1   #以1秒截断音频

path = r'D:\student\隐写分析\数据\TIMIT'
files = os.listdir(path)
files = [path + '\\' + f for f in files if f.endswith('.wav')]
print(len(files))
#
# def SetFileName(waveFileName):
#     for i in range(len(files)):
#         FileName = files[i]
#         print('SetFileName File Name is:',FileName)
#         FileName = waveFileName
#
# def CutFile():
#     for i in range(len(files)):
#         FileName = files[i]
#         print('CutFile Name is:',FileName)
#         f = wave.open(r""+FileName,'rb')
#         params = f.getparams()
#         nchannels, sampwidth, framerate, nframes= params[:4]
#         CutFrameNum = framerate * CutTimeOf
#         print('CutFrameNum=%d'%(CutFrameNum))
#         print("nchannels=%d" % (nchannels))
#         print("sampwidth=%d" % (sampwidth))
#         print("framerate=%d" % (framerate))
#         print("nframes=%d" % (nframes))
#         str_data = f.readframes(nframes)
#         f.close()
#
#         wave_data =  np.fromstring(str_data,dtype=np.short)
#         wave_data.shape = -1, 2
#         wave_data = wave_data.T
#         temp_data = wave_data.T
#
#         StepNum = CutFrameNum
#         StepTotalNum = 0
#         haha = 0
#
#         while StepTotalNum < nframes:
#             print('Step=%d'%(haha))
#             FileName = r'D:\Python\ML_Pro\audio\data\result_1s\\' + files[i][-17:-4] + '-'  + str(haha+1) + '.wav'
#             print(FileName)
#             temp_dataTemp = temp_data[StepNum * (haha):StepNum*(haha + 1)]
#             haha = haha + 1
#             StepTotalNum = StepNum * haha
#             temp_dataTemp.shape = 1,-1
#             temp_dataTemp = temp_dataTemp.astype(np.short)
#             f = wave.open(FileName,'wb')
#             # 配置声道数、采样率、量化位数
#             f.setnchannels(nchannels)
#             f.setframerate(framerate)
#             f.setsampwidth(sampwidth)
#             # 将wave_data转换成二进制数据写入文件
#             f.writeframes(temp_dataTemp.toString())
#             f.close()
# if __name__ == 'main':
#     CutFile()
# print('RunOver')