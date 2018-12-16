import librosa
import numpy as np
import os
import wave
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from svm import *
from svmutil import *

maxSignalNum = 1000

def second_order_derivative(signal):
    return np.array([signal[i+1] - 2*signal[i] - signal[i-1] for i in range(len(signal)-2) if i>=1])

def get_feature(singal, sr, isStego=1):
    mfccs = []
    labels = []
    for i in range(maxSignalNum):
        y = second_order_derivative(singal[i])
        mfcc = librosa.feature.mfcc(y=y, sr=sr[i], n_mfcc=24)  # (24,32)
        mfcc = np.mean(mfcc, axis=0).transpose()
        mfccs.append(mfcc)  # feature shape (32,)
        labels.append(isStego)
    return np.array(mfccs), np.array(labels)

def fetech_signal(input_file,isStego=1):
    files = os.listdir(input_file)
    files = [input_file + '\\' + f for f in files if f.endswith('.wav')]
    signals = []
    srs = []
    for i in range(maxSignalNum):
        y, sr = librosa.load(files[i], sr=None, duration=1)    # origin smaplerate:16k channels:1  duration:3.968
        signals.append(y)
        srs.append(sr)
    return np.array(signals), np.array(srs)

input_file_stego = r'D:\student\隐写分析\数据\1second_stego_data'
input_file_cover = r'D:\student\隐写分析\数据\1second_cover_data'
# y 就是音频的信号  duration=2读取前两秒的音频
# y, sr = librosa.load(r'D:\student\隐写分析\数据\stego_data_lsb\1200_stego.wav', sr=None,duration=2)    # origin smaplerate:16k channels:1  duration:3.968
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=512)  #  (24,2591)
# print(np.shape(np.array(mfcc).reshape(1,-1)))
stego_signals, stego_srs = fetech_signal(input_file_stego)
cover_singals , cover_srs = fetech_signal(input_file_cover,)

stego_mfccs,stego_labels = get_feature(stego_signals, stego_srs)  # (num_sample,32)
cover_mfccs, cover_labels = get_feature(cover_singals, cover_srs,-1)

# 训练数据集  (对训练和测试数据集进行合并)
X_data = np.row_stack((stego_mfccs,cover_mfccs))
y_data = np.append(stego_labels,cover_labels)

X_data, y_data = shuffle(X_data, y_data,random_state=7)
# 分离训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=5)
X_train = list([list(row) for row in X_train])
X_test = list([list(row) for row in X_test])
y_train = list(y_train)
y_test = list(y_test)

problem = svm_problem(y_train, X_train)
param = svm_parameter('-t 1 -c 4 -b 1')
model = svm_train(problem,param)

p_label, p_accuracy, p_validate = svm_predict(y_test, X_test, model)
print(p_label)
