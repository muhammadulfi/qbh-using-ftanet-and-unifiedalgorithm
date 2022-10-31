import os
import sys
import time
import pickle
import argparse
import datetime
import warnings
import numpy as np
import tensorflow as tf

sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from FTANet.network.ftanet_reverse import create_regularized_model, create_model
from FTANet.generator import create_data_generator
from FTANet.loader import load_data, load_data_for_test #TODO
from FTANet.evaluator import evaluate

batch_size = 4
seg_len = 128
in_shape = (seg_len, 320, 3)
loss_type = 'categorical_crossentropy'

valid_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining_r\\pl_valid_file_2007.pkl'
# model_file = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet\model\\retrain\model_2007_r-epoch8.h5'

# valid_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining_r\\pd2_valid_file.pkl'
model_file = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet\model\\retrain_r\\rds_ftamodel-epoch50.h5'
with open(valid_file, 'rb') as f:
    xlist, ylist = pickle.load(f)
    
# model = create_model(input_shape=in_shape)
model = create_regularized_model(factor = 0.0001, rate = 0.5, input_shape=in_shape)
model.load_weights(model_file)
model.compile(loss=loss_type, optimizer=(SGD(learning_rate=0.001))) 
avg_eval_arr = evaluate(model, xlist, ylist, batch_size)
print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
    avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4]))

# VR 78.99% VFA 2.54% RPA 69.65% RCA 73.24% OA 77.00% reverse adam non regularized - pl_valid_file_2007 file
# VR 81.93% VFA 2.70% RPA 71.42% RCA 76.11% OA 78.22% reverse adam non regularized - pd2_valid_file file
# VR 92.90% VFA 8.42% RPA 71.16% RCA 76.14% OA 75.92% reverse regularized sgd - pl_valid_file_2007 file
# VR 92.91% VFA 7.83% RPA 70.64% RCA 76.12% OA 75.67% reverse regularized sgd - pd2_valid_file file
# VR 93.34% VFA 8.37% RPA 71.01% RCA 77.74% OA 75.80% epoch 51