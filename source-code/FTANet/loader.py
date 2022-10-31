import os
import sys
import time
import pickle
import pyprind
import numpy as np
import tensorflow as tf
sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')

from numpy import genfromtxt
from FTANet.cfp import getTimeArr
from FTANet.cfp import cfp_process

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
        bar = pyprind.ProgBar(len(feature_files), track_time=False)
        for fname in feature_files:
            # get filepath             
            fname = fname.replace("\n","")
            fname = fname.split(", ")
            
            feature = np.load(fname[0]) 
            
            pitch = genfromtxt(fname[1], delimiter=',') 
            pitch = pitch[1:,1]              
            pitch = np.insert(pitch, 0, 0)
            pitch = np.insert(pitch, -1, pitch[-1])
            
            ## Transfer to mapping
            CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
            
            # make pitch to nearest cen_freq
            for index in range(len(pitch)):
                if pitch[index] != 0:
                    pos = (np.abs(CenFreq-pitch[index])).argmin()        
                    pitch[index] = CenFreq[pos]
            
            mapping = seq2map(pitch, CenFreq) # one-hot representation for label
            
            ## Crop to segments
            xlist, ylist, num = batchize(feature, mapping, xlist, ylist, seg_len)
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
        bar = pyprind.ProgBar(len(feature_files), track_time=False)
        for fname in feature_files:
            # get filepath             
            fname = fname.replace("\n","")
            fname = fname.split(", ")
            
            feature = np.load(fname[0])
            
            pitch = genfromtxt(fname[1], delimiter=',') 
            pitch = pitch[1:,1]              
            pitch = np.insert(pitch, 0, 0)
            pitch = np.insert(pitch, -1, pitch[-1])
            time_arr = np.arange(0.01, 8, 0.01)
            ref_arr = np.concatenate((time_arr[:, None], pitch[:, None]), axis=1)
            
            data = batchize_test(feature, seg_len)
            xlist.append(data)
            ylist.append(ref_arr[:, :])
            bar.update()

        dataset = (xlist, ylist)
        with open(data_file, 'wb') as f:
            print('Saving data...', end="\r")
            pickle.dump(dataset, f)
            print("Saved {} segments to {}".format(len(xlist), data_file))
    
    return xlist, ylist

# For Test
if __name__ == '__main__':
    # train_x, train_y, train_num = load_data('/data1/project/MCDNN/data/train_npy.txt')
    # print(train_y[0].shape)
    # for batch_y in train_y:
    #     for i in range(batch_y.shape[1]):
    #         y = np.argmax(batch_y[:, i])
    #         if y!=0:
    #             print(y)
    def est(output, CenFreq, time_arr):
        # output: (freq_bins, T)
        CenFreq[0] = 0
        est_time = time_arr
        est_freq = np.argmax(output, axis=0)

        for j in range(len(est_freq)):
            est_freq[j] = CenFreq[int(est_freq[j])]

        if len(est_freq) != len(est_time):
            new_length = min(len(est_freq), len(est_time))
            est_freq = est_freq[:new_length]
            est_time = est_time[:new_length]

        est_arr = np.concatenate((est_time[:, None], est_freq[:, None]), axis=1)

        return est_arr

    list_file = '/data1/project/MCDNN/data/test_02_npy.txt'
    # _, ylist = load_data_for_test(list_file) # test this func #Okay
    with open(list_file) as f:
        feature_files = f.readlines()
    data_folder = list_file[:-len(list_file.split('/')[-1])]
    # print(datapath)

    fname = feature_files[0]
    fname = fname.replace('.npy', '').rstrip()
    ref_arr = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')
    # ref_arr = ylist[0]

    CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
    mapping = seq2map(ref_arr[:, 1], CenFreq) # (321, T)
    est_arr = est(mapping, CenFreq, ref_arr[:, 0])

    from evaluator import melody_eval
    eval_arr = melody_eval(ref_arr, est_arr)
    print(eval_arr)

    cnt = 0
    for i in range(min(np.shape(est_arr)[0], np.shape(ref_arr)[0])):
        if est_arr[i][1] != ref_arr[i][1]:
            cnt += 1
    print(cnt)
    

