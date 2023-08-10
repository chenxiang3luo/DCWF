#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from sklearn import preprocessing
import numpy as np
import torch
from util.sample import CategoriesSampler,RandomIdentitySampler
import pdb
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
def get_matrix(y_test, predicted_labels):
    cnf_matrix = confusion_matrix(y_test, predicted_labels)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/((TP+FN+1e-8))
    # Specificity or true negative rate
    TNR = TN/(TN+FP+1e-8) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP+1e-8)
    # print(TP)
    # print((TP+FP))
    # Negative predictive value
    NPV = TN/(TN+FN+1e-8)
    # Fall out or false positive rate
    FPR = FP/(FP+TN+1e-8)
    # False negative rate
    FNR = FN/(TP+FN+1e-8)
    # False discovery rate
    FDR = FP/(TP+FP+1e-8)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    F1 = 2*PPV*TPR/(PPV+TPR+1e-8)

    return TPR,FPR,F1
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


def get_base_dataloader(dataset):

    train_dataset = np.load(f'/root/datasets/FSCIL/{dataset}.npz',allow_pickle=True)
    X_train = train_dataset['data']
    y_train = train_dataset['labels']
    enc = MyLabelEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)

    train_set = myDataset(X_train,y_train)

    trainloader = DataLoader(dataset=train_set, batch_size=512, shuffle=True,
                                              num_workers=16)
    
    return trainloader

def get_dataloader(args,session):

    trainset, trainloader, testloader = get_new_dataloader(args,session)
    return trainset, trainloader, testloader


def get_base_dataloader_meta(args,dataset):

    train_dataset = np.load(f'/root/datasets/FSCIL/{dataset}.npz',allow_pickle=True)
    x = train_dataset['data']
    y = train_dataset['labels']
    y = [str(lab) for lab in y]
    enc = MyLabelEncoder()
    enc.fit(y)
    y = enc.transform(y)
    print(x.shape)
    
    train_set = myDataset(x,y) 
    num = args.shot+args.query

    sampler = RandomIdentitySampler(labels=y,batch_size=args.way*num,num_instances=num,max_iters=8000)
    data_loader = DataLoader(train_set,batch_sampler=sampler,num_workers=16)


    return data_loader


def get_new_dataloader(args,data_set,num_class):
    file_name = '/root/datasets/FSCIL/'+data_set+'.npz'
    train_novel = np.load(file_name,allow_pickle=True)
    train_data = train_novel['data']
    train_labels = train_novel['labels']
    train_labels = [str(i) for i in train_labels]
    enc = MyLabelEncoder()
    enc.fit(train_labels)
    train_labels = enc.transform(train_labels)

    classes = np.unique(train_labels)
    support_data_novel = []
    support_labels_novel = []
    query_data_novel = []
    query_labels_novel = []

    for novel_class in classes:
        inds = np.argwhere(train_labels==novel_class)


        support_samples = inds[:args.shot].reshape(-1)
        query_samples = inds[args.shot:args.shot+args.query].reshape(-1)

        support = np.array(train_data[support_samples])
        support_label = train_labels[support_samples]
        query = np.array(train_data[query_samples])
        query_label = np.array(train_labels[query_samples])

        support_data_novel.extend(support)
        support_labels_novel.extend(support_label)

        query_data_novel.extend(query)
        query_labels_novel.extend(query_label)

    support_data_novel = np.array(support_data_novel)
    support_labels_novel = np.array(support_labels_novel)
    print(support_data_novel.shape)

    query_data_novel = np.array(query_data_novel)
    query_labels_novel = np.array(query_labels_novel)
    print(query_labels_novel.shape)

    train_set = myDataset(support_data_novel,support_labels_novel) 
    support_loader = DataLoader(dataset=train_set, batch_size = 4 , shuffle=True,
                                                  num_workers=args.num_workers)

    test_set = myDataset(query_data_novel,query_labels_novel)
    query_loader = DataLoader(dataset=test_set, batch_size = test_set.__len__(), shuffle=False,
                                             num_workers=args.num_workers)

    return support_loader, query_loader


def get_open_dataloader(session,args):
    open_size = 9000
    train_dataset_fw = np.load('/root/Lab/train_data_fw.npz',allow_pickle=True)
    test_dataset_fw = np.load('/root/Lab/test_data_fw.npz',allow_pickle=True)
    test_dataset_open = np.load('/root/dataset/dataset/tor_open_400000w.npz',allow_pickle=True)
    open_data = test_dataset_open['data']
    open_labels = test_dataset_open['labels']
    open_sample = np.random.permutation(open_labels.shape[0])[:open_size]
    open_data = open_data[open_sample]
    open_labels = open_labels[open_sample]+999

    if(session==0):
        train_data = train_dataset_fw['data'][:args.base_class*args.shot]
        train_labels = train_dataset_fw['labels'][:args.base_class*args.shot]
    else:
        train_data = train_dataset_fw['data'][(args.base_class+args.way*(session-1))*args.shot:(args.base_class+args.way*session)*args.shot]
        train_labels = train_dataset_fw['labels'][(args.base_class+args.way*(session-1))*args.shot:(args.base_class+args.way*session)*args.shot]
    test_data = test_dataset_fw['data'][:(args.base_class+args.way*session)*args.query]
    test_labels = test_dataset_fw['labels'][:(args.base_class+args.way*session)*args.query]
    
    train_set = myDataset(train_data,train_labels)

    # always load entire dataset in one batch    
    trainloader = DataLoader(dataset=train_set, batch_size=train_set.__len__() , shuffle=False,
                                                  num_workers=args.num_workers)

    test_data = np.concatenate([test_data,open_data])
    test_labels = np.concatenate([test_labels,open_labels])
    testset = myDataset(test_data,test_labels)

    testloader = DataLoader(dataset=testset, batch_size=args.batch_size_inference, shuffle=False,
                                             num_workers=args.num_workers)

    return train_set, trainloader, testloader
