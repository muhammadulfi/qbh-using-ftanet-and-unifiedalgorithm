import time
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from FTANet_melodic.network.ftanet import create_model
from FTANet_melodic.cfp import cfp_process, getTimeArr
from FTANet_melodic.loader import get_CenFreq, seq2map, batchize, batchize_test
from FTANet_melodic.evaluator import est, iseg, std_normalize

# load model
model_path = './FTANet_melodic/model/retrain/model_2007-epoch11.h5'
# model_path = './FTANet_melodic/model/retrain/praatlabel_model_2007-epoch31.h5'
in_shape = (320, 128, 3)
model = create_model(input_shape=in_shape)
model.load_weights(model_path)

def ftanet_extraction(model, query, batch_size = 320):
    preds = []
    data, CenFreq, time_arr = cfp_process(query, model_type='vocal', sr=8000, hop=80, window=768)
    feature = batchize_test(data, 128)
    
    for i in range(len(feature)):
        x = feature[i]
        
        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j*batch_size : ]
                length = x.shape[0]-j*batch_size
            else:
                X = x[j*batch_size : (j+1)*batch_size]
                length = batch_size

            X = np.reshape(X, (1, 320, 128, 3))
            prediction = model.predict(X, length)
            preds.append(prediction)
            
    preds = np.concatenate(preds, axis=0)
    preds = iseg(preds)

    # trnasform to f0ref
    # est_arr = est(preds, CenFreq, time_arr)
    
    est_arr = est(preds, CenFreq, time_arr)
    f0 = []
    mel_time = []
    for x in est_arr:
        mel_time.append(x[0])
        f0.append(x[1])
    
    return mel_time, f0  

# ftanet extraction
datasets = ['mirqbsh', 'ioacas']
output_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\\result_ftanet\\new_retrain'
# output_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\\result_ftanet\\new_retrain_praatlabel'

for dataset in datasets:
    if dataset == 'ioacas':
        with open(".\database\MIR-QBSH\querywav_list.txt",'r') as f:
            querylist = f.readlines()
        f.close()
    else:
        with open(".\database\IOACAS_QBH\querywav_list.txt",'r') as f:
            querylist = f.readlines()
        f.close()
    print(querylist)








def single_song_extraction(filepath, model_path, model_type='vocal', db=None, output=None):
    # preparation
    preds = []
    batch_size = 320
    in_shape = (320, 128, 3)
    
    if output is not None:
        output_file = output
        filename = filepath
    else :
        if db == 'MIR-QBSH':
            file = filepath.split("\\")
            filename = '-'.join(file[-3:])
            output_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\\ftanet_result\mir-qbsh', filename + ".csv")
            output_file = output_file.replace(".wav","")
            output_file = output_file.replace("\n","")    
    
    print('Load the model for song extraction.....')
    start = time.time()
    model = create_model(input_shape=in_shape)
    model.load_weights(model_path)
    end = time.time()
    print('Model load in {} sec!'.format(round(end-start,3)))
    
    start = time.time()
    data, CenFreq, time_arr = cfp_process(filepath, model_type='vocal', sr=8000, hop=80, window=768)
    print("CFP Feature Extraction is done, begin to predict the f0....")
    
    feature = batchize_test(data, 128)
    for i in range(len(feature)):
        x = feature[i]
        
        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j*batch_size : ]
                length = x.shape[0]-j*batch_size
            else:
                X = x[j*batch_size : (j+1)*batch_size]
                length = batch_size

            X = np.reshape(X, (1, 320, 128, 3))
            prediction = model.predict(X, length)
            preds.append(prediction)
            
    preds = np.concatenate(preds, axis=0)
    preds = iseg(preds)

    # trnasform to f0ref
    # est_arr = est(preds, CenFreq, time_arr)
    
    est_arr = est(preds, CenFreq, time_arr)
    f0 = []
    mel_time = []
    for x in est_arr:
        mel_time.append(x[0])
        f0.append(x[1])
    # save to csv
    # ypred = pd.DataFrame(est_arr)
    # ypred.to_csv(output_file, index=False)
    pd.DataFrame({'time':mel_time, 'f0':f0}).to_csv(output_file, index=False)
    end = time.time()
    print('{} extraction is done in {} and saved to {}'.format(filename +'.wav', round(end-start, 3), output_file))


def multiple_song_extraction(filelist, model_type, model):
    print('test')