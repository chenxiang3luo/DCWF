# from keras.preprocessing.sequence import pad_sequences
# import threading
from os.path import join
import traceback
from tqdm import tqdm
import numpy as np
#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from sklearn import preprocessing
import numpy as np
import torch
import pdb
from torch.utils.data import Dataset, DataLoader
# from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self
class myDataset(Dataset):
    def __init__(self,x,y):
        x = x.astype(np.float32)
        y = torch.tensor(y,dtype = torch.long)
        self.x_data = x.reshape(x.shape[0],1,x.shape[1])
        self.y_data = y
        self.len = x.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

def set_up_datasets():
    return myDataset()


def get_dataloader():
    train_dataset = np.load('/root/datasets/FSCIL/tor_900w_2500tr.npz',allow_pickle=True)
    tran = MyLabelEncoder()
 
    train_dataset_data = train_dataset['data']
    train_dataset_labels = train_dataset['labels']
    tran.fit(train_dataset_labels)
    train_dataset_labels = tran.transform(train_dataset_labels)
    train_data,test_data,train_label,test_label = train_test_split(train_dataset_data,train_dataset_labels,test_size=0.045,stratify=train_dataset_labels,shuffle=True)
    print(len(np.unique(test_label)))
    train_set = myDataset(train_data,train_label)
    trainloader = DataLoader(dataset=train_set, batch_size=256, shuffle=True,
                                              num_workers=16)
    testset = myDataset(test_data,test_label)

    testloader = DataLoader(dataset=testset, batch_size=len(testset), shuffle=False,
                                             num_workers=16)
    return trainloader, testloader


