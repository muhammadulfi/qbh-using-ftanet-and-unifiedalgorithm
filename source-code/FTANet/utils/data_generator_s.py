import os
import sys
import time
import pickle
import pyprind
import numpy as np
import pandas as pd

from numpy import genfromtxt

def get_CenFreq(StartFreq=80, StopFreq=1000, NumPerOct=48):
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            if i == 0 :
                central_freq.append(0)
            else:
                central_freq.append(round(CenFreq, 3))
        else:
            break
    return central_freq

def seq2map(seq, CenFreq):
    CenFreq[0] = 0
    gtmap = np.zeros((len(seq), len(CenFreq)))
    for i in range(len(seq)):
        v = np.where(CenFreq == seq[i])
        gtmap[i, v[0][0]] = 1
    return gtmap 

def batchize(data, gt, xlist, ylist, size=430):
    # menghilangkan error perbedaan shape
    if data.shape[-1] != gt.shape[0]:
        new_length = min(data.shape[-1], gt.shape[0])

        data = data[:, :, :new_length]
        gt = gt[:new_length, :]
        
    num = int(gt.shape[0] / size)
    if gt.shape[0] % size != 0:
        num += 1
    for i in range(num):
        if (i + 1) * size > gt.shape[0]:
            batch_x = np.zeros((data.shape[0], data.shape[1], size))
            batch_y = np.zeros((size, gt.shape[-1]))
            
            tmp_x = data[:, :, i * size:]
            tmp_y = gt[i * size:, :]            
            
            batch_x[:, :, :tmp_x.shape[-1]] += tmp_x 
            batch_y[:tmp_y.shape[0], :] += tmp_y
            xlist.append(batch_x.transpose(2, 1, 0))
            ylist.append(batch_y)
            break
        else:
            batch_x = data[:, :, i * size:(i + 1) * size]
            batch_y = gt[i * size:(i + 1) * size, :]        
            xlist.append(batch_x.transpose(2, 1, 0))
            ylist.append(batch_y)

    return xlist, ylist, num

def batchize_test(data, size=430):
    xlist = []
    num = int(data.shape[-1] / size)
    if data.shape[-1] % size != 0:
        num += 1
    for i in range(num):
        if (i + 1) * size > data.shape[-1]:
            batch_x = np.zeros((data.shape[0], data.shape[1], size))

            tmp_x = data[:, :, i * size:]

            batch_x[:, :, :tmp_x.shape[-1]] += tmp_x
            xlist.append(batch_x.transpose(2, 1, 0))
            break
        else:
            batch_x = data[:, :, i * size:(i + 1) * size]
            xlist.append(batch_x.transpose(2, 1, 0))

    return np.array(xlist)


def load_data(list_file, seg_len=430):
    data_file = list_file.split('/')[-1].replace('.txt', '_s.pkl')
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            print('Saved data found! loading data...', end="\r")
            start = time.time()
            xlist, ylist = pickle.load(f)
            end = time.time()
            print('Load saved data successfully in {} sec'.format(round(end-start, 3)), end="\r")

    else:
        # get file list
        with open(list_file) as f:
            feature_files = f.readlines()

        xlist = []
        ylist = []
        bar = pyprind.ProgBar(len(feature_files), track_time=True)
        for fname in feature_files:
            # get filepath             
            fname = fname.replace("\n","")
            fname = fname.split(", ")
            
            feature = np.load(fname[0]) 
            
            # label
            label = pd.read_csv(fname[1])
            pitch =  np.asarray(label['f0'])
            time_label =  np.asarray(label['time'])
            
            # get feature with same time axis
            time_index = np.arange(0.01, 8, 0.01)
            time_index = list(np.around(time_index ,2))
            
            index_lbl = []
            time_index = list(time_index)
            for i in range(len(time_label)):
                index_lbl.append(time_index.index(time_label[i]))

            feature_short = np.zeros((feature.shape[0], feature.shape[1], len(index_lbl)))
            for i in range(len(index_lbl)):
                feature_short[:,:,i] = feature[:,:,index_lbl[i]]

            ## Transfer to mapping
            CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
            
            # make pitch to nearest cen_freq
            for index in range(len(pitch)):
                if pitch[index] != 0:
                    pos = (np.abs(CenFreq-pitch[index])).argmin()        
                    pitch[index] = CenFreq[pos]
            
            mapping = seq2map(pitch, CenFreq) # one-hot representation for label
            
            ## Crop to segments
            xlist, ylist, num = batchize(feature_short, mapping, xlist, ylist, seg_len)
            bar.update()

        dataset = (xlist, ylist)
        with open(data_file, 'wb') as f:
            print('Saving data...', end="\r")
            pickle.dump(dataset, f)
            print("Saved {} segments to {}".format(len(xlist), data_file))
    
    return xlist, ylist, len(ylist)

def load_data_for_test(list_file, seg_len=430):
    data_file = list_file.split('/')[-1].replace('.txt', '.pkl')
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            print('Saved data found! loading data...', end="\r")
            start = time.time()
            xlist, ylist = pickle.load(f)
            end = time.time()
            print('Load saved data successfully in {} sec'.format(round(end-start, 3)), end="\r")

    else:
        # get file list
        with open(list_file) as f:
            feature_files = f.readlines()

        xlist = []
        ylist = []
        bar = pyprind.ProgBar(len(feature_files), track_time=True)
        for fname in feature_files:
            # get filepath             
            fname = fname.replace("\n","")
            fname = fname.split(", ")
            
            feature = np.load(fname[0])
            
            # label
            label = pd.read_csv(fname[1])
            pitch =  np.asarray(label['f0'])
            time_label =  np.asarray(label['time'])
            
            # get feature with same time axis
            time_index = np.arange(0.01, 8, 0.01)
            time_index = list(np.around(time_index ,2))
            
            index_lbl = []
            time_index = list(time_index)
            for i in range(len(time_label)):
                index_lbl.append(time_index.index(time_label[i]))

            feature_short = np.zeros((feature.shape[0], feature.shape[1], len(index_lbl)))
            for i in range(len(index_lbl)):
                feature_short[:,:,i] = feature[:,:,index_lbl[i]]

            ref_arr = np.concatenate((time_label[:, None], pitch[:, None]), axis=1)
            
            data = batchize_test(feature_short, seg_len)
            xlist.append(data)
            ylist.append(ref_arr[:, :])
            bar.update()

        dataset = (xlist, ylist)
        with open(data_file, 'wb') as f:
            print('Saving data...', end="\r")
            pickle.dump(dataset, f)
            print("Saved {} segments to {}".format(len(xlist), data_file))
    
    return xlist, ylist

train_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining_r\\sd_train_file.txt'
valid_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining_r\\sd_valid_file.txt'

print('Loading data...')
start = time.time()
train_x, train_y, train_num = load_data(train_path, seg_len=128)
end = time.time()
print('Loaded {} segments from {} in {} sec.'.format(train_num, train_path, round(end-start, 3)))

start = time.time()
valid_x, valid_y  = load_data_for_test(valid_path, seg_len=128)
end = time.time()
print('Loaded features for {} files from {} in {} sec.'.format(len(valid_y), valid_path, round(end-start, 3)))

print(np.shape(train_x), np.shape(train_y))
print(np.shape(valid_x), np.shape(valid_y))