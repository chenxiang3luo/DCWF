#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from sklearn import preprocessing
import numpy as np
import torch
from util.sample import CategoriesSampler
from util.sample import *
from torch.utils.data import Dataset, DataLoader
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


def get_base_dataloader():

    train_dataset = np.load('/root/datasets/FSCIL/AWF_775.npz',allow_pickle=True)
    x = train_dataset['data']
    y = train_dataset['labels']
    # train_dataset_500 = np.load('/root/datasets/FSCIL/KNN_FRONT.npz',allow_pickle=True)
    # base_data = train_dataset_500['data']
    # y_labels = train_dataset_500['labels']
    # X_train,X_test, y_train, y_test =train_test_split(base_data,y_labels,test_size=25*100, random_state=42,shuffle=True,stratify=y_labels)
    # x = X_test
    # y = y_test
    enc = MyLabelEncoder()
    enc.fit(y)
    y = enc.transform(y)
    print(x.shape)
    
    train_set = myDataset(x,y)

    sampler = RandomIdentitySampler(labels=y,batch_size=128,num_instances=8,max_iters=8000)
    data_loader = DataLoader(train_set,batch_sampler=sampler,num_workers=16)


    return data_loader

def get_dataloader(args,session):

    trainset, trainloader, testloader = get_new_dataloader(args,session)
    return trainset, trainloader, testloader


def get_base_dataloader_meta(args,do_augment=True):
    train_dataset_500 = np.load('/root/FSCIL/dataset/train_data_DF_front_base.npz',allow_pickle=True)
    x = train_dataset_500['data']
    y = train_dataset_500['labels']
    # enc=MyLabelEncoder()
    # enc.fit(train_dataset_500['labels'])#训练LabelEncoder,将电脑，手表，手机编码为0,1,2
    # y = enc.transform()

    # uniques,ids = np.unique(train_dataset_500['labels'],return_inverse=True)
    # y = to_categorical(ids, len(uniques))
    
    X_train,X_test, y_train, y_test =train_test_split(x,y,test_size=0.1, random_state=7,shuffle=True,stratify=y)
    print(X_train.shape)
    train_set = myDataset(X_train,y_train)
    test_set = myDataset(X_test,y_test)


    sampler = CategoriesSampler(train_set.y_data, args.max_train_iter, args.num_ways_training,
                                 args.num_shots_training + args.num_query_training)

    trainloader =DataLoader(dataset=train_set, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = DataLoader(
        dataset=test_set, batch_size=args.batch_size_inference, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_set, trainloader, testloader

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
    support_loader = DataLoader(dataset=train_set, batch_size = 4, shuffle=True,num_workers=args.num_workers)

    test_set = myDataset(query_data_novel,query_labels_novel)
    query_loader = DataLoader(dataset=test_set, batch_size = 256, shuffle=False,
                                             num_workers=args.num_workers)

    return support_loader,query_loader


def get_new_included_dataloader(args,data_set,num_classes,train_included):
    file_name = '/root/datasets/FSCIL/AWF_775_TrainingIncluded.npz'
    train_novel = np.load(file_name,allow_pickle=True)
    train_data = train_novel['data']
    train_data = np.array(train_data)
    train_labels = train_novel['labels']
    train_labels = [str(lab) for lab in train_labels]
    train_labels = np.array(train_labels)
    classes = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    # class_sample = np.random.permutation(classes.shape[0])
    classes = classes[:train_included]
    support_data_novel = []
    support_labels_novel = []
    query_data_novel = []
    query_labels_novel = []
    for novel_class in classes:
        inds = np.argwhere(train_labels==novel_class)
        # samples = np.random.permutation(len(inds))

        # support_samples = samples[]
        support_samples = inds[:args.shot].reshape(-1)

        # query_samples = samples[]
        query_samples = inds[args.shot:args.shot+args.query].reshape(-1)

        support_data_novel.extend(train_data[support_samples])
        support_labels_novel.extend(train_labels[support_samples])

        query_data_novel.extend(train_data[query_samples])
        query_labels_novel.extend(train_labels[query_samples])
    print(len(query_labels_novel))

    file_name = '/root/datasets/FSCIL/'+data_set+'.npz'
    train_novel = np.load(file_name,allow_pickle=True)
    train_data = train_novel['data']
    train_data = np.array(train_data)
    train_labels = train_novel['labels']
    train_labels = [str(lab) for lab in train_labels]
    train_labels = np.array(train_labels)
    classes = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    # class_sample = np.random.permutation(classes.shape[0])
    classes = classes[:num_classes-train_included]
    for novel_class in classes:
        inds = np.argwhere(train_labels==novel_class)
        # samples = np.random.permutation(len(inds))

        support_samples = inds[:args.shot].reshape(-1)
        # support_samples = inds[support_samples].reshape(-1)

        # query_samples = samples[args.shot:args.shot+args.query]
        query_samples = inds[args.shot:args.shot+args.query].reshape(-1)

        support_data_novel.extend(train_data[support_samples])
        support_labels_novel.extend(train_labels[support_samples])

        query_data_novel.extend(train_data[query_samples])
        query_labels_novel.extend(train_labels[query_samples])

    print(len(query_labels_novel))
    support_data_novel = np.array(support_data_novel)
    support_labels_novel = [str(label) for label in support_labels_novel]
    enc = MyLabelEncoder()
    enc.fit(support_labels_novel)
    support_labels_novel = enc.transform(support_labels_novel)
    print(support_labels_novel.shape)

    query_data_novel = np.array(query_data_novel)
    # fit(num) will cause problem
    query_labels_novel = [str(label) for label in query_labels_novel]
    enc = MyLabelEncoder()
    enc.fit(query_labels_novel)
    query_labels_novel = enc.transform(query_labels_novel)
    print(query_labels_novel.shape)

    train_set = myDataset(support_data_novel,support_labels_novel) 
    support_loader = DataLoader(dataset=train_set, batch_size = 4 , shuffle=True,
                                                  num_workers=args.num_workers)

    test_set = myDataset(query_data_novel,query_labels_novel)
    query_loader = DataLoader(dataset=test_set, batch_size = 256, shuffle=False,
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


