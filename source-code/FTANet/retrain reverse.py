import os
import sys
import time
import argparse
import datetime
import warnings
import numpy as np
import tensorflow as tf

sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_accuracy
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
# sess = tf.compat.v1.Session(config=config) 
# tf.compat.v1.disable_v2_behavior()
# K.set_session(sess)

# tf.config.experimental.set_lms_enabled(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from network.ftanet_reverse import create_regularized_model as create_model

from generator import create_data_generator
from loader import load_data, load_data_for_test #TODO
from evaluator import evaluate

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# batch_size = 16
batch_size = 4
seg_len = 128
in_shape = (seg_len, 320, 3)
epochs = 15
patience = 10
loss_type = 'categorical_crossentropy'
short = False

checkpoint_model_file = 'E:\\Kuliah\\S2\\Thesis\\source code thesis\\FTANet\\model\\retrain_r\\rda_ftamodel.h5'
# if loss_type == 'binary_crossentropy':
#     checkpoint_model_file = checkpoint_model_file.replace('.h5', 'b.h5')
# elif loss_type == 'categorical_crossentropy':
#     checkpoint_model_file = checkpoint_model_file.replace('.h5', 'c.h5')

log_file_name = checkpoint_model_file.replace('model\\retrain_r\\', 'log\\').replace('.h5', '.txt')
log_file = open(log_file_name, 'w')

# add message to log
def log(message):
    # message_bytes = '{}\n'.format(message).encode(encoding='utf-8') 
    ct = datetime.datetime.now()   
    message_bytes = '{} - {}\n'.format(ct, message)
    log_file.write(message_bytes)
    print(message)

train_list_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining_r\\pd2_train_file.txt'
valid_list_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining_r\\pd2_valid_file.txt'

beginprocess = datetime.datetime.now()   
log('Loading data...')
start = time.time()
train_x, train_y, train_num = load_data(train_list_file, seg_len=128)
end = time.time()
log('Loaded {} segments from {} in {} sec.'.format(train_num, train_list_file, round(end-start, 3)))

start = time.time()
valid_x, valid_y  = load_data_for_test(valid_list_file, seg_len=128)
end = time.time()
log('Loaded features for {} files from {} in {} sec.'.format(len(valid_y), valid_list_file, round(end-start, 3)))

endprocess = datetime.datetime.now()   
log('Load data is start at {}'.format(beginprocess))
log('Process is finished at {}'.format(endprocess))
print(np.shape(train_x), np.shape(train_y), np.shape(valid_x), np.shape(valid_y))
# train_dataset = tf.data.Dataset.from_tensor_slices((np.asarray(train_x), np.asarray(train_y)))
# test_dataset = tf.data.Dataset.from_tensor_slices((np.asarray(valid_x), np.asarray(valid_y)))
# print(train_dataset)
# print(type(train_dataset))
# exit()

log('Creating generators...')
train_generator = create_data_generator(train_x, train_y, batch_size)

log('Creating model...')
# model_file = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet\model\\retrain_r\\rds_ftamodel-epoch51.h5'
model = create_model(factor = 0.00001, rate = 0.5, input_shape=in_shape)
# model.load_weights(model_file)
model.compile(loss=loss_type, optimizer=(Adam(learning_rate=0.0001))) 

begin_training = datetime.datetime.now() 
log('Training start at {}...'.format(begin_training))
epoch, iteration = 0, 0
best_OA, best_epoch = 0, 0
mean_loss = 0
time_start = time.time()
while epoch < epochs:
    iteration += 1
    print('Epoch {}/{} - {:3d}/{:3d}'.format(
        epoch+1, epochs, iteration%(train_num//batch_size), train_num//batch_size), end='\r')
    
    X, y = next(train_generator)
    
    loss = model.train_on_batch(X, y)
    mean_loss += loss
    
    if iteration % (train_num//batch_size) == 0:
        ## train meassage
        epoch += 1
        traintime  = time.time() - time_start
        mean_loss /= train_num//batch_size
        print('', end='\r')
        log('Epoch {}/{} - {:.1f}s - loss {:.4f}'.format(epoch, epochs, traintime, mean_loss))
        
        ## valid results
        avg_eval_arr = evaluate(model, valid_x, valid_y, batch_size, short)
        log('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}% BestOA {:.2f}%'.format(
            avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4], best_OA))
        # save to model
        if avg_eval_arr[-1] > best_OA:
            best_OA = avg_eval_arr[-1]
            best_epoch = epoch
            checkpoint_epoch_weightmodel_file = checkpoint_model_file.replace('.h5', '_weight-epoch'+str(epoch)+'.h5') 
            checkpoint_epoch_model_file = checkpoint_model_file.replace('.h5', '-epoch'+str(epoch)+'.h5') 
            model.save_weights(checkpoint_epoch_weightmodel_file)
            model.save(checkpoint_epoch_model_file)
            log('Saved to ' + checkpoint_epoch_model_file)
        
        # early stopping
        if epoch - best_epoch >= patience:
            log('Early stopping at {} epoch with best OA {:.2f}%'.format(epoch, best_OA))
            break
        ## initialization
        mean_loss = 0
        time_start = time.time()

end_training = datetime.datetime.now() 
log('Training end at {}...'.format(end_training))
log_file.close()
