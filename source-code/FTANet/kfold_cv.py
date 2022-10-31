import os
import sys
import argparse
import time
import datetime
import pickle
import numpy as np
import tensorflow as tf
sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from sklearn.model_selection import KFold
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import optimizers

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def standardize(train):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)+0.000001

    X_train = (train - mean) / std
    return X_train

test = 'spec'
if test == 'spec': 
    from FTANet.network.ftanet_mod1c import create_model, create_regularized_model 
    train_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining_spec\pd4_train_file.pkl'
    in_shape = (32, 513, 1)
else:
    from FTANet.network.ftanet_reverse import create_model 
    train_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining_r\pl_train_file_2007.pkl'
    in_shape = (128, 320, 3)

with open(train_path, 'rb') as f:
    xlist, ylist = pickle.load(f)

if test != 'spec':
    xlist = np.asarray(xlist)
    ylist = np.asarray(ylist)

xlist = standardize(xlist)

exit()

num_folds = 5
batch_size = 8
no_epochs = 5

kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
acc_per_fold = []
loss_per_fold = []
for train, test in kfold.split(xlist, ylist):
    
    if test == 'spec': 
        model = create_regularized_model(factor = 0.0001, rate = 0.5, input_shape=in_shape)
    else:
        model = create_model(input_shape=in_shape)
        
    # optimizer = Adam(clipvalue=0.5, learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=(Adam(clipvalue=0.5, learning_rate=0.0001)), metrics=['accuracy'])
    
    print('------------------------------------------------------------------------')
    print('Training for fold {} ...'.format(fold_no))
    
    history = model.fit(xlist[train], ylist[train],
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=1)
    
    # Generate generalization metrics
    scores = model.evaluate(xlist[test], ylist[test], verbose=0)
    try:
        print('Score for fold {}: {} of {}; {} of {}%'.format(fold_no, model.metrics_names[0], scores[0], model.metrics_names[1], scores[1]*100))
    except:
        print(scores)
    print('done training')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
    
acc_avg, loss_avg = 0, 0
for i in range(len(acc_per_fold)):
    print('Score for fold {}: {}\t{}'.format(i+1, acc_per_fold[i], loss_per_fold[i]))
    acc_avg += acc_per_fold[i]
    loss_avg += loss_per_fold[i]
print('Avg for all fold is {}\t{}'.format(acc_avg/num_folds, loss_avg/num_folds))