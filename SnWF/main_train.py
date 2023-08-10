#10-cv deep fingerprinting code
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from model.model_keras import *
import numpy as np
import time
import random
import os
from tqdm import tqdm
import tensorflow as tf
from util.data_loader import *
from keras.layers import Input
from util.keras_callbacks import *
from sklearn.model_selection import StratifiedShuffleSplit
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 20} ) 
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

if __name__ == "__main__":
    random.seed(7)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    EXP_Type = 'OpenWorld_NoDef'
    # print ("Experimental Type: ", EXP_Type)
    # network and training
    NB_EPOCH = 100
    # print ("Number of Epoch: ", NB_EPOCH)
    BATCH_SIZE = 128
    VERBOSE = 1
    LENGTH = 5000
    # collecting after the sixth epochs
    

    NB_CLASSES = 100 # number of outputs: 100 Monitored websites + 1 Unmonitored websites
    INPUT_SHAPE = (505,1)
    model = DFNet.build_BASE(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    snapshot = SnapshotCallbackBuilder(6,100,6,0.002)
    X_train,X_test, y_train, y_test = get_feature_dataloader()
    OPTIMIZER = tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, decay=0.004)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
            metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=BATCH_SIZE,epochs=NB_EPOCH,validation_split = 0.1,callbacks=snapshot.get_callbacks(model_prefix='snWF'))
    # checkpoint_path = './dae_model_weight.ckpt'
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                              save_weights_only=True,
    #                                              verbose=1)
#     X_train,X_test, y_train, y_test = get_base_dataloader()
# #     X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
# #     print(X_train[0])
#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#     model = DFNet.build_DAE(input_shape=INPUT_SHAPE)
#     OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
#     model.compile(loss="mse", optimizer=OPTIMIZER,
#             metrics=None)
#     model.fit(X_train, X_train, batch_size=BATCH_SIZE,epochs=NB_EPOCH,validation_split = 0.1,verbose = VERBOSE,callbacks=[callback])
#     model.save_weights('./dae_weights')
    

