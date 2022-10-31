import time
import os
import pyprind
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from network.ftanet_reverse import create_regularized_model as create_model
from cfp import cfp_process, getTimeArr
from loader import get_CenFreq, seq2map, batchize, batchize_test
from evaluator import est

def ftanet_extraction(model, query, batch_size = 128):
    preds = []
    data, CenFreq, time_arr = cfp_process(query, model_type='vocal', sr=8000, hop=80, window=768)
    feature = batchize_test(data, batch_size)
    
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

            X = np.reshape(X, (1, 128, 320, 3))
            prediction = model.predict(X, length)
            preds.append(prediction)
            
    preds = np.concatenate(np.concatenate(preds, axis=0), axis=0)
    
    est_arr = est(preds, CenFreq, time_arr)
    f0 = []
    mel_time = []
    for x in est_arr:
        mel_time.append(x[0])
        f0.append(x[1])
    
    return mel_time, f0  


# load model
# in_shape = (128, 320, 3)
# loss_type = 'categorical_crossentropy'
# model_file = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet\model\\retrain_r\\rds_ftamodel-epoch51.h5'
# model = create_model(factor = 0.00001, rate = 0.5, input_shape=in_shape)
# model.load_weights(model_file)

models_used = ['rda']
in_shape = (128, 320, 3)
computation_time = []

for model_used in models_used:
    if model_used == 'rc':
        model_path = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet\model\\retrain\model_2007_r-epoch8.h5'
    elif model_used == 'rda' :
        model_path = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet\model\\retrain_r\\rda_ftamodel-epoch2.h5'
    else :
        model_path = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet\model\\retrain\model_2007_rb-epoch2.h5'
    model = create_model(factor = 0.00001, rate = 0.5, input_shape=in_shape)
    model.load_weights(model_path)

    # ftanet extraction
    datasets = ['mir-qbsh', 'ioacas']    
    # datasets = ['mir-qbsh']
    # output_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\\result_ftanet\\{}'.format(model_used)
    

    for dataset in datasets:
        if dataset == 'mir-qbsh':
            with open(".\database\MIR-QBSH\querywav_list.txt",'r') as f:
                querylist = f.readlines()
            f.close()
        else:
            with open(".\database\IOACAS_QBH\querywav_list.txt",'r') as f:
                querylist = f.readlines()
            f.close()
        count = 1
        totaltime = 0
        bar = pyprind.ProgPercent(len(querylist), track_time=True, title='Melody Extraction {} for {}'.format(model_used, dataset))
        for query in querylist:        
            query = query.replace("\n","")
            # prepare the output file
            file = query.split("\\")
            if dataset == 'mir-qbsh':
                filename = '-'.join(file[-3:])
            else:
                filename = file[-1]
            
            output_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\\result_ftanet\\'+ model_used +'\\'+ dataset +'\\'+ filename + ".csv")
            output_file = output_file.replace(".wav","")
            
            start = time.time()
            mel_time, f0 = ftanet_extraction(model, query)
            end = time.time()
            pd.DataFrame({'time':mel_time, 'f0':f0}).to_csv(output_file, index=False)
            # print('Melody extraction saved at {} in {:.3f} sec'.format(output_file, round(end-start, 3)))
            totaltime += round(end-start, 3)
            # print('\n{}/{} Melody Extracted'.format(count, len(querylist)), end='\r')
            count += 1
            bar.update()
        avg_time = totaltime/len(querylist) 
        computation_time.append([totaltime, avg_time])
        print(totaltime, avg_time)
        

for i in range(len(datasets)):
    print(datasets[i])
    print("Total time for melody extraction is {:.3f} sec".format(computation_time[i][0]))
    print("Avg time for melody extraction is {:.3f} sec".format(computation_time[i][1]))
    
# for i in range(len(datasets)):
#     print(datasets[i], ' - fix_epoch12')
#     print("Total time for melody extraction is {:.3f} sec".format(computation_time[i+2][0]))
#     print("Avg time for melody extraction is {:.3f} sec".format(computation_time[i+2][1]))