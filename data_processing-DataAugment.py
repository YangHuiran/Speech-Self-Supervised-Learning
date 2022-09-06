#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa.display
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import torchaudio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import IPython.display as ipd
import cv2




# In[3]:


def pitch_shifting(x, sr, n_steps, bins_per_octave=12):
    # sr: 音频采样率
    # n_steps: 要移动多少步
    # bins_per_octave: 每个八度音阶(半音)多少步
    return librosa.effects.pitch_shift(x, sr, n_steps, bins_per_octave=bins_per_octave)


# In[3]:


outer_path = ('/Users/yanghuiran/Graduate-VS/data/dev-clean')
folderlist = os.listdir(outer_path)
print(folderlist)

image_shape = (224, 224)

idx = 1
for folder in folderlist:
    if not folder.startswith('.'):

        sub_path = os.path.join(outer_path,folder)
        subfolderlist = os.listdir(sub_path)

        for sub_folder in subfolderlist:
            print("sub_folder",sub_folder)

            if not sub_folder.startswith('.'):

                file_path = os.path.join(sub_path,sub_folder)    
                #print('file_path===',file_path)
                filelist = os.listdir(file_path)


                for file in filelist:
                    #print('file = ',file)

                    if file.endswith('.flac'):
                        path = os.path.join(file_path,file)

                        wav, sample_rate = librosa.load(os.path.join(file_path,file))
                        
#                         #时间偏移
#                         start_ = int(np.random.uniform(-4800,4800))
                        
#                         if start_ >= 0:
#                             wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001,0.001, start_)]
#                         else:
#                             wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), wav[:start_]]
                        
                        #速度变化
                        
                        speed = random.uniform(0.5,2)
                        speed_rate = np.random.uniform(speed)
                        wav_speed = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()

                        # if len(wav_speed_tune) < 16000:

                        #     pad_len = 16000 - len(wav_speed_tune)
                        #     wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                        #                            wav_speed_tune,
                        #                            np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
                        # else: 
                        #     cut_len = len(wav_speed_tune) - 16000
                        #     wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+16000]
#                         ipd.Audio(wav_speed_tune, rate=sample_rate, autoplay=True)
                        #音频增强
                        wav_frequency = pitch_shifting(wav_speed, sr=16000, n_steps=4, bins_per_octave=12)
                        
                        samples = np.random.uniform(low=-0.2, high=0.2, size=wav_frequency.shape).astype(np.float32)
#                         ipd.Audio(samples,rate=16000, autoplay=True)
                        #调节音量，混合噪音
                        wav_final = wav_frequency * np.random.uniform(0.8, 1.4) + samples * np.random.uniform(0, 0.01)
#                         ipd.Audio(wav_with_bg, rate=sample_rate, autoplay=True)

                        

                #         sample, sample_rate = torchaudio.load(os.path.join(big_file,file))
                #         mel_transfrom = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                #                                                                 n_fft=n_fft,
                #                                                                 win_length=win_length,
                #                                                                 hop_length=hop_length,
                #                                                                 center=True,
                #                                                                 pad_mode="reflect",
                #                                                                 power=2.0,
                #                                                                 norm='slaney',
                #                                                                 onesided=True,
                #                                                                 n_mels=n_mels,
                #                                                                 mel_scale="htk",)

                #         original_mel = mel_transfrom(sample)

                #         plot_spectrogram(original_mel[0], title="Original")

                        #image.save('/Users/yanghuiran/graduate/train_dataset/{:0>6d}.jpg'.format(idx))



                        window_size = 1024
                        window = np.hanning(window_size)
                        stft  = librosa.core.spectrum.stft(wav_final, n_fft=window_size, hop_length=512, window=window)
                        out = 2 * np.abs(stft) / np.sum(window)

                        # For plotting headlessly

                        fig = plt.Figure()
                        canvas = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        # p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
                        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)

                        filename = '{:0>6d}.jpg'.format(idx)
                        path = os.path.dirname('/Users/yanghuiran/Graduate-VS/data/dev_FreAndTempoAndNoisy/')

                        fig.savefig(os.path.join(path, filename))
                        idx = idx + 1





# In[18]:


wav, sample_rate = librosa.load("/Users/yanghuiran/Graduate-VS/data/augmentationTest/test.flac")
# ipd.Audio(data, rate=sample_rate, autoplay=True)
#音频增强
# wav_new = pitch_shifting(wav, sr=16000, n_steps=6, bins_per_octave=12)
# ipd.Audio(wav_frequencyChanged, rate=sample_rate, autoplay=True)
#速度变化
speed = random.uniform(0.1,2)
print(speed)
speed_rate = np.random.uniform(speed)
print(speed_rate)
wav_final = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
 #调节音量，混合噪音
# samples = np.random.uniform(low=-0.2, high=0.2, size=wav_speed.shape).astype(np.float32)
# wav_final = wav_speed * np.random.uniform(0.8, 1.4) + samples * np.random.uniform(0, 0.02)

ipd.Audio(wav_final, rate=sample_rate, autoplay=True)


# In[ ]:





# In[ ]:





# In[26]:


D = np.abs(librosa.stft(wav_final)) ** 2  # stft频谱
S = librosa.feature.melspectrogram(S=D)  # 使用stft频谱求Mel频谱
plt.figure(figsize=(8, 5))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),fmax=8000)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
plt.savefig("/Users/yanghuiran/Graduate-VS/data/augmentationTest/sample.png", bbox_inches = "tight", pad_inches = 0.0)


# fig = plt.Figure()
# canvas = FigureCanvas(fig)
# ax = fig.add_subplot(111)
# # p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
# p = librosa.display.specshow(librosa.amplitude_to_db(wav_final, ref=np.max), ax=ax)
# path = os.path.dirname('/Users/yanghuiran/Graduate-VS/data/augmentationTest/')
# fig.savefig(os.path.join(path, "sample"))


# In[20]:


fs = 16000
plt.figure(figsize=(7, 5))
plt.title("Spectrogram")
plt.specgram(wav_final, Fs=8000, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('Time')
plt.ylabel('Hz')
plt.tight_layout()
plt.show()


# In[7]:


plt.figure(figsize=(7, 5))
plt.title("Waveform")
time = np.arange(0, len(wav_new)) * (1.0 / fs)
plt.plot(time, wav_new)
plt.xlabel('Time')
plt.ylabel('Hz')

plt.tight_layout()
plt.show()


# In[8]:


# ########### 画图
fs = 16000
plt.subplot(2, 2, 1)
plt.title("语谱图", fontsize=15)
plt.specgram(wav_frequencyChanged, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.subplot(2, 2, 2)
plt.title("波形图", fontsize=15)
time = np.arange(0, len(wav_frequencyChanged)) * (1.0 / fs)
plt.plot(time, wav_frequencyChanged)
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('振幅', fontsize=15)

plt.tight_layout()
plt.show()

