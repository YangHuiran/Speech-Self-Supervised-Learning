#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import torchaudio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import codecs
import csv

# BASE_DIR = '/Users/yanghuiran/Graduate-VS/data/'
# train_folder = BASE_DIR+'train_dataset20220721/'
# train_annotation = BASE_DIR+'annotated_train_data/'

# files_in_train = sorted(os.listdir(train_folder))
# files_in_annotated = sorted(os.listdir(train_annotation))

# images=[i for i in files_in_train if i in files_in_annotated]
#images=[i for i in files_in_train]

df = pd.DataFrame()
df.to_csv("/Users/yanghuiran/Graduate-VS/data/largedatatest.csv",index=False)

# df.to_csv("/Users/yanghuiran/Graduate-VS/data/fineturn_test/female_labels.csv",index=False)
# df.to_csv("/Users/yanghuiran/Graduate-VS/data/female_dir.csv",index=False)
# df.to_csv("/Users/yanghuiran/Graduate-VS/data/male_dir.csv",index=False)
print("stop")

#df['images']=[train_folder+str(x) for x in images]
#df['labels']=[train_annotation+str(x) for x in images]
#males' data name
val_males = ["61","260","672","908","1089","1188","1320","2300","2830","4077","5105","5639","6930","7021","7127","7176","7729","8224","8230","8455"]
test_males = ["8098","8108","8226","8419","8425","8580","8609","8629","8630","8474","8770","8797","8838","","","","","","","","",]
train_males=["26","27","60","78","87","118","163","196","201","229","233","254","307","311","374","405","412","445","446","458","460","481","625","831","839","909","1034","1040","1081","1235","1334"]
idx = 1
outer_path = ('/Users/yanghuiran/Graduate-VS/data/pretrain_notused')
folderlist = os.listdir(outer_path)
for folder in folderlist:
    if not folder.startswith('.'):
        
        print('folder=',folder)
        sub_path = os.path.join(outer_path,folder)
        print('sub_path=',sub_path)
        subfolderlist = os.listdir(sub_path)
        print('subfolderlist=',subfolderlist)
        csvdata = [[]]

        for sub_folder in subfolderlist:

            if not sub_folder.startswith('.'):

                file_path = os.path.join(sub_path,sub_folder)    
                #print('file_path===',file_path)
                filelist = os.listdir(file_path)


                for file in filelist:
                    #print('file = ',file)

                    if file.endswith('.flac'):
                        startname = file[0:file.find("-")]
                        
                        print(file)
                        path = os.path.join(file_path,file)
                        sample, sample_rate = librosa.load(os.path.join(file_path,file))

                        window_size = 1024
                        window = np.hanning(window_size)
                        stft  = librosa.core.spectrum.stft(sample, n_fft=window_size, hop_length=512, window=window)
                        out = 2 * np.abs(stft) / np.sum(window)

                        fig = plt.Figure()
                        canvas = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        # p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
                        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)

                        filename = '{:0>6d}.jpg'.format(idx)
                        
                        path = os.path.dirname('/Users/yanghuiran/Graduate-VS/data/largedatatest.csv/')

                        fig.savefig(os.path.join(path, filename))

                        series = [os.path.join(path, filename)]
#                         csv_data = pd.DataFrame([series])

#                             path = "/Users/yanghuiran/Graduate-VS/data/male_lables.csv"
                        with open(r'/Users/yanghuiran/Graduate-VS/data/fineturn_test/largedatatest.csv',mode='a',newline='',encoding='utf8') as file_csv:
#                             file_csv = codecs.open(path, 'w+', 'utf-8')  
                            writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(series)
                        
#                         if startname in test_males:
#                             label = '1'
                            
#                             path = os.path.dirname('/Users/yanghuiran/Graduate-VS/data/fineturn_test/male_test/')

#                             fig.savefig(os.path.join(path, filename))

#                             series = [os.path.join(path, filename), label]
#     #                         csv_data = pd.DataFrame([series])

# #                             path = "/Users/yanghuiran/Graduate-VS/data/male_lables.csv"
#                             with open(r'/Users/yanghuiran/Graduate-VS/data/fineturn_test/male_labels.csv',mode='a',newline='',encoding='utf8') as file_csv:
#     #                             file_csv = codecs.open(path, 'w+', 'utf-8')  
#                                 writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#                                 writer.writerow(series)
#                         else:
#                             label = '0'
                            
#                             path = os.path.dirname('/Users/yanghuiran/Graduate-VS/data/fineturn_test/female_test/')

#                             fig.savefig(os.path.join(path, filename))

#                             series = [os.path.join(path, filename), label]
#     #                         csv_data = pd.DataFrame([series])

# #                             path = "/Users/yanghuiran/Graduate-VS/data/trainLabel.csv"
                           
#                             with open(r'/Users/yanghuiran/Graduate-VS/data/fineturn_test/female_labels.csv',mode='a',newline='',encoding='utf8') as file_csv:
#     #                             file_csv = codecs.open(path, 'w+', 'utf-8')  
#                                 writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#                                 writer.writerow(series)
                        

                        idx = idx + 1


# df.to_csv(r'/Users/yanghuiran/Graduate-VS/data/testLabel.csv', header=None)
pd.read_csv(r'/Users/yanghuiran/Graduate-VS/data/testLabel.csv')

