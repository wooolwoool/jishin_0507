# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:02:59 2019

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

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

#%% load data

data = pd.read_csv('data/div_train/train_0001.csv')
'''
メモ
'time_to_failure'が0になるCSV
train_006.csv

'''

#%%  data summay
def data_summary(data):
    print('col  : ', data.columns)
    print('shape: ', data.shape)
    

data_summary(data)

#%%  data plot

def data_plot(data, col_list, rang = []):
    if rang == []:
        for l in col_list:
            plt.plot(data.ix[:, l], '-')
            plt.show()
    else:
        for l in col_list:
            plt.plot(data.ix[rang, l], '-')
            plt.show()
            
def data_double_plot(data, col_list, rang = []):
    if not len(col_list)==2:
        print('colum length is not 2')
    else:
        if rang == []:
            fig, ax1 = plt.subplots()
            ax1.plot(data.ix[:, col_list[0]], 'b-')
            ax2 = ax1.twinx()
            ax2.plot(data.ix[:, col_list[1]], 'r-')
            plt.show()
        else:
            fig, ax1 = plt.subplots()
            ax1.plot(data.ix[rang, col_list[0]], 'b-')
            ax2 = ax1.twinx()
            ax2.plot(data.ix[rang, col_list[1]], 'r-')
            plt.show()
            
'''
rang指定しないとほぼ落ちるので注意
'''
col_list = ['acoustic_data', 'time_to_failure']
data_plot(data, col_list, range(4436000,4450000))
data_double_plot(data, col_list, range(4436000,4450000))

#%%   divide data by time_to_failre

csv_list = glob.glob('data/train/*')
data_cash_flag = 0
csv_num = 1
train_dir = 'data/div_train/'

for l in csv_list:
    tmp_data = pd.read_csv(l)
    diff = np.array(tmp_data.ix[1:, 'time_to_failure']) - np.array(tmp_data.ix[:len(tmp_data.index)-2, 'time_to_failure'])
    start_ind = np.where(diff > 0.1)[0]
    if data_cash_flag == 0:
        if not len(start_ind) == 0:
            data_cash = tmp_data.ix[:start_ind[0], :]
            data_cash.to_csv(train_dir + 'train_{:04}.csv'.format(csv_num))
            print('train_{:04}'.format(csv_num))
            csv_num += 1
            del data_cash
            gc.collect()
            data_cash = tmp_data.ix[:start_ind[0]+1, :]
            data_cash_flag = 1
        else:
            data_cash = tmp_data
            data_cash_flag = 1
    else:
        if not len(start_ind) == 0:
            data_cash = pd.concat([data_cash, tmp_data.ix[:start_ind[0], :]])
            data_cash.to_csv(train_dir + 'train_{:04}.csv'.format(csv_num))
            print('train_{:04}'.format(csv_num))
            csv_num += 1
            del data_cash
            gc.collect()
            data_cash = tmp_data.ix[start_ind[0]+1:, :]
        else:
            data_cash = pd.concat([data_cash, tmp_data])


