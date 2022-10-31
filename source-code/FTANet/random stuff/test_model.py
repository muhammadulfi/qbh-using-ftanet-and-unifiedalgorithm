import tensorflow as tf
import os
import sys
sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')

from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.metrics import categorical_accuracy
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
# sess = tf.compat.v1.Session(config=config) 
# tf.compat.v1.disable_v2_behavior()
# K.set_session(sess)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from FTANet.network.ftanet_reverse import create_model, create_regularized_model 
# from FTANet.network.ftanet_mod1c import create_model, create_regularized_model 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
batch_size = 8
seg_len = 32 # 513
# in_shape = (seg_len, 513, 1)
in_shape = (128, 320, 3)
epochs = 50
patience = 10

model = create_regularized_model(factor = 0.0001, rate = 0.5, input_shape=in_shape)
model.compile(loss='categorical_crossentropy', optimizer=(Adam(learning_rate=0.01)))
# model_path = 'E:\Kuliah\S2\Thesis\source code thesis\FTANet\\model\\retrain\model_2007_r-epoch8.h5'
# model.load_weights(model_path)

# model2 = create_model_mod1c(input_shape=in_shape)
# model2.compile(loss='binary_crossentropy', optimizer=(Adam(learning_rate=0.01)))


# print(model.summary())
# print('---------------------xxxx---------------------')
# print(model2.summary())
# print(model.loss)