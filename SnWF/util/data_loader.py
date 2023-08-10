import numpy as np
import torch
import pdb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
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

class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

def get_base_dataloader():


    train = np.load('./feature_2000.npy',allow_pickle=True).item()
    train_X ,train_y = train['feature'], train['label']
    # y=enc.fit_transform(train_dataset_500['labels'])#训练LabelEncoder,将电脑，手表，手机编码为0,1,2

    # uniques,ids = np.unique(train_dataset_500['labels'],return_inverse=True)
    # y = to_categorical(ids, len(uniques))
    
    X_train,X_test, y_train, y_test =train_test_split(train_X,train_y,test_size=0.1, random_state=0,shuffle=True,stratify=train_y)


    return X_train,X_test, y_train, y_test

def get_feature_dataloader():


    train = np.load('./feature.npy',allow_pickle=True).item()
    train_X ,train_y = train['feature'], train['label']
    # y=enc.fit_transform(train_dataset_500['labels'])#训练LabelEncoder,将电脑，手表，手机编码为0,1,2

    # uniques,ids = np.unique(train_dataset_500['labels'],return_inverse=True)
    # y = to_categorical(ids, len(uniques))
    
    X_train,X_test, y_train, y_test =train_test_split(train_X,train_y,test_size=0.1, random_state=0,shuffle=True,stratify=train_y)


    return X_train,X_test, y_train, y_test
def get_dataloader(session):
    # 全部的 数据
    train_dataset = np.load('/root/Lab/train_data_df.npz',allow_pickle=True)
    
    train_dataset_labels = train_dataset['labels']
    train_dataset_data = train_dataset['data']

    way = 20
    shot = 5
    base_class = 100
    train_num = base_class*2500+way*session*75
    train_data = train_dataset_data[:train_num]
    train_label = train_dataset_labels[:train_num]
    test_size=(base_class+way*session)*70
    
    train_data,test_data,train_label,test_label = train_test_split(train_data,train_label,test_size=test_size,stratify=train_label)
    print(len(np.unique(test_label)))
    train_set = myDataset(train_data,train_label)
    trainloader = DataLoader(dataset=train_set, batch_size=128, shuffle=True,
                                              num_workers=16)
    testset = myDataset(test_data,test_label)

    testloader = DataLoader(dataset=testset, batch_size=128, shuffle=False,
                                             num_workers=16)
    return trainloader, testloader

def get_new_dataloader(session,):
    # 很少的数据
    base_class = 100
    shot = 5
    way = 20
    query = 95
    train_name = '/root/Lab/train_data_fw.npz'
    test_name = '/root/Lab/test_data_fw.npz'
    train_dataset_fw = np.load(train_name,allow_pickle=True)
    test_dataset_fw = np.load(test_name,allow_pickle=True)
    train_data = train_dataset_fw['data'][:(base_class+way*session)*shot]
    train_labels = train_dataset_fw['labels'][:(base_class+way*session)*shot]

    test_data = test_dataset_fw['data'][:(base_class+way*session)*query]
    test_labels = test_dataset_fw['labels'][:(base_class+way*session)*query]
    # Load support set (don't do data augmentation here )
    print(np.unique(train_labels).shape)
    train_set = myDataset(train_data,train_labels)

    # always load entire dataset in one batch    
    trainloader = DataLoader(dataset=train_set, batch_size=base_class+way*session, shuffle=False,
                                                  num_workers=16)


    # test on all encountered classes
    # class_new = get_session_classes(args, session)
    
    testset = myDataset(test_data,test_labels)

    testloader = DataLoader(dataset=testset, batch_size=128, shuffle=False,
                                             num_workers=16)

    return trainloader, testloader