# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:30:55 2019

"""

# import関連
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import datetime
import math
import glob
import gc
import random
import scipy.stats

from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten 
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D

'''
Conv1Dを使用してモデルを作成する。
X:data_lenの長さの振動波形
Y:Xのtime_to_failureの平均値
'''

#%%  
file_num = 1 # number of read data
data_len = int(150000/100) # data length

#%%  data preprosess
'''
所持しているデータからfile_numだけロードし
data_lenの長さに区切る
'''
csv_list = glob.glob('data/div_train/*')
#random.shuffle(csv_list)
origin_data_X = []
origin_data_Y = []

for l in csv_list[:file_num]:
    tmp_data = np.array(pd.read_csv(l))
    for i in range(int(tmp_data.shape[0]/data_len)):
        origin_data_X.append(tmp_data[i*data_len: (i+1)*data_len, 1])
        origin_data_Y.append(tmp_data[i*data_len: (i+1)*data_len, 2].mean())

random.shuffle(origin_data_X)
random.shuffle(origin_data_Y)

train_num = int(len(origin_data_X)*0.8) 

train_data_X = scipy.stats.zscore(np.array(origin_data_X[:train_num]).reshape((-1, data_len, 1)))
train_data_Y = scipy.stats.zscore(np.array(origin_data_Y[:train_num]).reshape((-1, 1)))
test_data_X = scipy.stats.zscore(np.array(origin_data_X[train_num+1:]).reshape((-1, data_len, 1)))
test_data_Y = scipy.stats.zscore(np.array(origin_data_Y[train_num+1:]).reshape((-1, 1)))
print(train_data_Y.shape)
print(test_data_Y.shape)

#%%  model difine

model = Sequential()
model.add(Conv1D(12, 1000, padding='same', input_shape=(data_len, 1), activation='relu'))
model.add(MaxPooling1D(10, padding='same'))
model.add(Conv1D(12, 100, padding='same', activation='relu'))
model.add(MaxPooling1D(10, padding='same'))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()

#%% model learning

history = model.fit(train_data_X, train_data_Y, validation_split=0.1, epochs=100, 
                    batch_size=20, validation_data=(test_data_X, test_data_Y))

#%% validation
    plt.plot(range(epochs), history.history['loss'], label='loss')
    #plt.plot(range(epochs), history.history['acc'], label='acc')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend() 
    plt.show()



