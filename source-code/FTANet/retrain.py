import os
import argparse
import time
import datetime
import warnings
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_accuracy
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.disable_v2_behavior()
K.set_session(sess)

# tf.config.experimental.set_lms_enabled(True)

from network.ftanet import create_model

from generator import create_data_generator
from loader import load_data, load_data_for_test #TODO
from evaluator import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# checkpoint_model_file = './FTANet_melodic/model/model_file_smalltraining.h5'
checkpoint_model_file = 'E:\\Kuliah\\S2\\Thesis\\source code thesis\\FTANet_melodic\\model\\retrain\\model_2007.h5'
log_file_name = checkpoint_model_file.replace('model\\retrain\\', 'log\\').replace('.h5', '.txt')
# print(log_file_name)
log_file = open(log_file_name, 'w')
# add message to log
def log(message):
    # message_bytes = '{}\n'.format(message).encode(encoding='utf-8') 
    ct = datetime.datetime.now()   
    message_bytes = '{} - {}\n'.format(ct, message)
    log_file.write(message_bytes)
    print(message)

# batch_size = 16
batch_size = 8
seg_len = 128
in_shape = (320, seg_len, 3)
epochs = 50
patience = 10

# train_list_file = './database/retrain_list/train_file.txt'
# valid_list_file = './database/retrain_list/valid_file.txt'
# train_list_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining\\train_file_github_fix.txt'
# valid_list_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining\\valid_file_github_fix.txt'
train_list_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining\\train_file_2007.txt'
valid_list_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining\\valid_file_2007.txt'
beginprocess = datetime.datetime.now()   
log('Loading data...')
start = time.time()
train_x, train_y, train_num = load_data(train_list_file, seg_len=128)
# print(np.shape(train_x), np.shape(train_y))
end = time.time()
log('Loaded {} segments from {} in {} sec.'.format(train_num, train_list_file, round(end-start, 3)))
start = time.time()
valid_x, valid_y = load_data_for_test(valid_list_file, seg_len=128)
end = time.time()
log('Loaded features for {} files from {} in {} sec.'.format(len(valid_y), valid_list_file, round(end-start, 3)))

endprocess = datetime.datetime.now()   
log('Load data is start at {}'.format(beginprocess))
log('Process is finished at {}'.format(endprocess))

# exit()

log('Creating generators...')
train_generator = create_data_generator(train_x, train_y, batch_size=8)

log('Creating model...')

# model_file_github_fix_epoch12.h5 retrain\SGD_model_2007f-epoch2.h5
model = create_model(input_shape=in_shape)
model.compile(loss='binary_crossentropy', optimizer=(Adam(learning_rate=0.01))) 
# model_path = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet_melodic\model\\model_file_github_fix_epoch12.h5'
# model_path = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet_melodic\model\\ftanet_a.h5'
# model.load_weights(model_path)

# start = time.time()
# print('model loaded, start to evaluate.....')
# avg_eval_arr = evaluate(model, valid_x, valid_y, 16)
# end = time.time()
# log('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
#             avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4]))
# print('in ', round(end-start, 3), ' sec')
# exit()
# log_file.close() 
begin_training = datetime.datetime.now() 
log('Training start at {}...'.format(begin_training))
epoch, iteration = 0, 0
best_OA, best_epoch = 0, 0
mean_loss = 0
time_start = time.time()
while epoch < epochs:
    iteration += 1
    print('Epoch {}/{} - {:3d}/{:3d}'.format(
        epoch+1, epochs, iteration%(train_num//16), train_num//16), end='\r')
    # 取1个batch数据
    X, y = next(train_generator)
    # 训练1个iteration
    loss = model.train_on_batch(X, y)
    mean_loss += loss
    # 每个epoch输出信息
    if iteration % (train_num//16) == 0:
        ## train meassage
        epoch += 1
        traintime  = time.time() - time_start
        mean_loss /= train_num//16
        print('', end='\r')
        log('Epoch {}/{} - {:.1f}s - loss {:.4f}'.format(epoch, epochs, traintime, mean_loss))
        ## valid results
        avg_eval_arr = evaluate(model, valid_x, valid_y, 16)
        # save to model
        if avg_eval_arr[-1] > best_OA:
            best_OA = avg_eval_arr[-1]
            best_epoch = epoch
            checkpoint_epoch_weightmodel_file = checkpoint_model_file.replace('.h5', '_weight-epoch'+str(epoch)+'.h5') 
            checkpoint_epoch_model_file = checkpoint_model_file.replace('.h5', '-epoch'+str(epoch)+'.h5') 
            model.save_weights(checkpoint_epoch_weightmodel_file)
            model.save(checkpoint_epoch_model_file)
            log('Saved to ' + checkpoint_epoch_model_file)
        log('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}% BestOA {:.2f}%'.format(
            avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4], best_OA))
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
