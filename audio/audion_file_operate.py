import numpy as np
import librosa
import matplotlib.pyplot as plt
import audioread
# from presets import Preset
import librosa.display
# 1.get the file path to include audio example


filename = r'D:\student\隐写分析\TIMIT\1.wav'
y,sr = librosa.load(filename,sr=44100)

melspec = librosa.feature.melspectrogram(y,sr,n_fft=1024,hop_length=512,n_mels=128)
logmelspec = librosa.power_to_db(melspec)

plt.figure()
plt.subplot(2,1,1)
librosa.display.waveplot(y=y,sr=sr)
plt.title('Beat waveform')

plt.subplot(2,1,2)
librosa.display.specshow(logmelspec,sr=sr,x_axis='time',y_axis='mel')
plt.title('Mel Spectrogram')
plt.tight_layout()  # 保证图不重叠
plt.show()

# with audioread.audio_open(filename) as fr:
#     print(fr.channels,fr.samplerate,fr.duration)



# y, sr = librosa.load(filename,duration=5,offset=35)
# M = _librosa.feature.melspectrogram(y=y)
# M_higher = librosa.feature.melspectrogram(y=y,hop_length=512)
#
# plt.figure(figsize=(6,6))
# ax = plt.subplot(3,1,1)
# _librosa.display.specshow(_librosa.power_to_db(M,ref=np.max),y_axis='mel',x_axis='time')
# plt.title('44100/1024/4096')
#
# plt.subplot(3,1,2,sharex=ax,sharey=ax)
# _librosa.display.specshow(_librosa.power_to_db(M_higher,ref=np.max),y_axis='mel',x_axis='time')
# plt.title('44100/512/4096')
#
# librosa['sr']=11025
# y2, sr2 = _librosa.load(filename,duration=5,offset=35)
# M2 = _librosa.feature.melspectrogram(y=y2,sr=sr2)
# plt.subplot(3,1,3,sharex=ax,sharey=ax)
# _librosa.display.specshow(_librosa.power_to_db(M2,ref=np.max),y_axis='mel',x_axis='time')
# plt.title('11025/1024/4096')
# plt.tight_layout()
# plt.show()


# 2.Load the audio as a waveform y‘
# Store the sampling rate as sr
# y, sr = librosa.load(filename)
# print(np.shape(y))
# plt.figure(figsize=(12,8))
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)),ref=np.max)
# plt.subplot(4,2,1)
# librosa.display.specshow(D,y_axis='linear')
# plt.colorbar(format='%+2.0f db')
# plt.title('Linear frequency power spectrogram')
#
# plt.subplot(4, 2, 2)
# librosa.display.specshow(D, y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Log-frequency power spectrogram')
#
# CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
# plt.subplot(4, 2, 3)
# librosa.display.specshow(CQT, y_axis='cqt_note')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Constant-Q power spectrogram (note)')
#
# plt.subplot(4, 2, 4)
# librosa.display.specshow(CQT, y_axis='cqt_hz')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Constant-Q power spectrogram (Hz)')
#
# C = librosa.feature.chroma_cqt(y=y, sr=sr)
# plt.subplot(4, 2, 5)
# librosa.display.specshow(C, y_axis='chroma')
# plt.colorbar()
# plt.title('Chromagram')
#
# plt.subplot(4, 2, 6)
# librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Linear power spectrogram (grayscale)')
#
# plt.subplot(4, 2, 7)
# librosa.display.specshow(D, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Log power spectrogram')
#
# plt.subplot(4, 2, 8)
# Tgram = librosa.feature.tempogram(y=y, sr=sr)
# librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
# plt.colorbar()
# plt.title('Tempogram')
# plt.tight_layout()
#
# plt.figure()
# tempo, beat_f = librosa.beat.beat_track(y=y, sr=sr, trim=False)
# beat_f = librosa.util.fix_frames(beat_f, x_max=C.shape[1])
# Csync = librosa.util.sync(C, beat_f, aggregate=np.median)
# beat_t = librosa.frames_to_time(beat_f, sr=sr)
# ax1 = plt.subplot(2,1,1)
# librosa.display.specshow(C, y_axis='chroma', x_axis='time')
# plt.title('Chroma (linear time)')
# ax2 = plt.subplot(2,1,2, sharex=ax1)
# librosa.display.specshow(Csync, y_axis='chroma', x_axis='time',
#                       x_coords=beat_t)
# plt.title('Chroma (beat time)')
# plt.tight_layout()
#
# plt.show()
# # 3.run the default beat tracker
# # tempo, beat_frames = rosa.beat.beat_track(y=y,sr=sr)
# #
# # print('Estimated tempo:{:.2f} beats per minute '.format(tempo))
# #
# # # 4.convert the frame indices of beat events to timestamps
# # beat_times = rosa.frames_to_time(beat_frames,sr=sr)
# #
# # print('Saving output to beat-times.csv',beat_times)
# # rosa.output.times_csv('beat-times.csv',beat_times)
# # set the hop length ;at length   ,512 samples ~=23ms
# hop_length =512
#
# # seperate harmonics and percussive into two waveform
# y_harmonic, y_percussive = rosa.effects.hpss(y)
# # beat track on the percussive signal
# tempo, beat_frames = rosa.beat.beat_track(y_percussive,sr=sr)
# # comput MFCC feature from the raw signal
# mfcc = rosa.feature.mfcc(y=y,sr=sr,hop_length=hop_length,n_mfcc=13)
# # And the first-order differences (delta features)
# mfcc_delta = rosa.feature.delta(mfcc)
# # stack and synchronize between beat events
# # This time, we will use the mean value(default) insteal of median
# beat_mfcc_delta = rosa.util.sync(np.vstack([mfcc,mfcc_delta]),beat_frames)
#
# # comput chroma feature from the harmonic signal
# chromgram = rosa.feature.chroma_cqt(y=y_harmonic,sr=sr)
#
# # Aggregate chroma features between events
# # We'll use the median value of each feature between beat frames
# beat_chroma = rosa.util.sync(chromgram,beat_frames,aggregate=np.median)
#
# # finally ,stack all beat-synchronous features together
# beat_features = np.vstack([beat_chroma,beat_mfcc_delta])
# print('shape of input:',sr)
# print('shape of chroma:',np.shape(beat_chroma))
# print('shape of mfcc_delta: ',np.shape(beat_mfcc_delta))
# print('shape of  vstack',np.shape(beat_features))
#
# # use audioread read audio file
# with audioread.audio_open(filename) as fr:
#     print(fr.channels, fr.samplerate,fr.duration)


