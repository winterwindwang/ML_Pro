#coding=utf-8
#测试文件
from sigprocess import *
from calcmfcc import *
import librosa
import numpy
import audioread

input_file = r'D:\student\隐写分析\数据\cover_data\2000.wav'
sig, rate = librosa.load(input_file,sr=None)
mfcc_feat = calcMFCC_delta_delta(sig,rate)
print(mfcc_feat.shape)

with audioread.audio_open(input_file) as f:
    print(f.duration,f.samplerate)