#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from sklearn import preprocessing
import numpy as np
import torch
from tqdm import tqdm
from util.sample import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
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

class myDataset_feature(Dataset):
    def __init__(self,x,y):
        x = x.astype(np.float32)
        y = torch.tensor(y,dtype = torch.long)
        self.x_data = x
        self.y_data = y
        self.len = x.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def set_up_datasets():
    return myDataset()


def get_base_dataloader(source_domain):

    train_dataset = np.load(f'/root/datasets/FSCIL/{source_domain}.npz',allow_pickle=True)
    x = train_dataset['data']
    y = train_dataset['labels']
    enc = MyLabelEncoder()
    enc.fit(y)
    y = enc.transform(y)
    print(x.shape)

    train_set = myDataset(x,y)
    
    sampler = RandomIdentitySampler(labels=y,batch_size=256,num_instances=8,max_iters=24000)
    data_loader = DataLoader(train_set,batch_sampler=sampler,num_workers=16)


    return data_loader

def get_dataloader(args,session):

    trainset, trainloader, testloader = get_new_dataloader(args,session)
    return trainset, trainloader, testloader



def get_meta_dataloader(args,data_set,num_class):
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

    for novel_class in tqdm(classes):
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
    support_loader = DataLoader(dataset=train_set, batch_size = 32, shuffle=True,num_workers=args.num_workers)

    test_set = myDataset(query_data_novel,query_labels_novel)
    query_loader = DataLoader(dataset=test_set, batch_size = 512, shuffle=False,
                                             num_workers=args.num_workers)

    return support_loader,query_loader

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

def get_meta_include_dataloader(args,data_set,num_class,train_included=0):
    file_name = '/root/datasets/FSCIL/AWF_775_TrainingIncluded.npz'
    train_novel = np.load(file_name,allow_pickle=True)
    train_data = train_novel['data']
    train_labels = train_novel['labels']
    train_labels = [str(i) for i in train_labels]
    train_labels = np.array(train_labels)
    
    classes = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    classes = classes[:train_included]
    support_data_novel = []
    support_labels_novel = []
    query_data_novel = []
    query_labels_novel = []

    for novel_class in tqdm(classes):
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
    classes = classes[:num_class-train_included]
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
    support_labels_novel = np.array(support_labels_novel)
    enc = MyLabelEncoder()
    enc.fit(support_labels_novel)
    support_labels_novel = enc.transform(support_labels_novel)
    print(support_data_novel.shape)

    query_data_novel = np.array(query_data_novel)
    query_labels_novel = np.array(query_labels_novel)
    enc = MyLabelEncoder()
    enc.fit(query_labels_novel)
    query_labels_novel = enc.transform(query_labels_novel)
    print(query_labels_novel.shape)

    train_set = myDataset(support_data_novel,support_labels_novel) 
    support_loader = DataLoader(dataset=train_set, batch_size = 32, shuffle=True,num_workers=args.num_workers)

    test_set = myDataset(query_data_novel,query_labels_novel)
    query_loader = DataLoader(dataset=test_set, batch_size = 256, shuffle=False,
                                             num_workers=args.num_workers)

    return support_loader,query_loader

def get_new_include_dataloader(args,feature_model,data_set,num_classes,train_included):
    file_name = '/root/datasets/FSCIL/AWF_775_TrainingIncluded.npz'
    train_novel = np.load(file_name,allow_pickle=True)
    train_data = train_novel['data']
    train_labels = train_novel['labels']
    train_labels = [str(lab) for lab in train_labels]
    train_labels = np.array(train_labels)
    base_means,base_data = get_base_feature('AWF_775_feature')
    classes = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    # class_sample = np.random.permutation(classes.shape[0])
    classes = classes[:train_included]
    support_data_novel = []
    support_labels_novel = []
    query_data_novel = []
    query_labels_novel = []
    for novel_class in tqdm(classes):
        inds = np.argwhere(train_labels==novel_class)
        # print(inds)
        support_samples = inds[:args.shot].reshape(-1)

        query_samples = inds[args.shot:args.shot+args.query].reshape(-1)

        support = np.array(train_data[support_samples])
        support = support.reshape(support.shape[0],1,support.shape[1])

        query = np.array(train_data[query_samples])
        query = query.reshape(query.shape[0],1,query.shape[1])

        support = torch.tensor(support,dtype=torch.float32).cuda()
        query = torch.tensor(query,dtype=torch.float32).cuda()

        feature_model.eval()
        with torch.no_grad():
            support = np.array(feature_model(support).cpu())
            query = np.array(feature_model(query).cpu())
        support_label = train_labels[support_samples]
        support,support_label = calibration_feture(support,support_label,base_means,base_data,args.shot,2)
    
        support_data_novel.extend(support)
        support_labels_novel.extend(support_label)

        query_data_novel.extend(query)
        query_labels_novel.extend(train_labels[query_samples])

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

    for novel_class in tqdm(classes):
        inds = np.argwhere(train_labels==novel_class)
        # print(inds)
        support_samples = inds[:args.shot].reshape(-1)

        query_samples = inds[args.shot:args.shot+args.query].reshape(-1)

        support = np.array(train_data[support_samples])
        support = support.reshape(support.shape[0],1,support.shape[1])

        query = np.array(train_data[query_samples])
        query = query.reshape(query.shape[0],1,query.shape[1])

        support = torch.tensor(support,dtype=torch.float32).cuda()
        query = torch.tensor(query,dtype=torch.float32).cuda()

        feature_model.eval()
        with torch.no_grad():
            support = np.array(feature_model(support).cpu())
            query = np.array(feature_model(query).cpu())
        support_label = train_labels[support_samples]
        support,support_label = calibration_feture(support,support_label,base_means,base_data,args.shot,2)
    
        support_data_novel.extend(support)
        support_labels_novel.extend(support_label)

        query_data_novel.extend(query)
        query_labels_novel.extend(train_labels[query_samples])

    support_data_novel = np.array(support_data_novel)
    support_labels_novel = np.array(support_labels_novel)
    enc = MyLabelEncoder()
    enc.fit(support_labels_novel)
    support_labels_novel = enc.transform(support_labels_novel)
    print(support_labels_novel.shape)

    query_data_novel = np.array(query_data_novel)
    query_labels_novel = np.array(query_labels_novel)
    enc = MyLabelEncoder()
    enc.fit(query_labels_novel)
    query_labels_novel = enc.transform(query_labels_novel)
    print(query_labels_novel.shape)

    train_set = myDataset_feature(support_data_novel,support_labels_novel) 
    support_loader = DataLoader(dataset=train_set, batch_size = 64 , shuffle=True,
                                                  num_workers=args.num_workers)

    test_set = myDataset_feature(query_data_novel,query_labels_novel)
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

def distribution_calibration(query, base_means, base_data, k=2,alpha=0.2):
    dist = []
    for i in range(len(base_means)):
        # dist.append(np.linalg.norm(query-base_means[i]))
        query = query
        base_mean = base_means[i]
        dot_product = np.dot(query, base_mean)
        # 计算向量的范数
        norm1 = np.linalg.norm(query)
        norm2 = np.linalg.norm(base_mean)
        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)
        dist.append(1-similarity)
    index = np.argpartition(dist, k)[:k]
    # if((1- dist[index[0]]) < 1):
    #     return None,None
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(query[np.newaxis, :], axis=0) 
    keys = [k for k in base_data.keys()]
    covs = []
    for ind in index:
        feature = np.array(base_data[keys[ind]])
        covs.append(np.cov(feature.T))
    calibrated_cov = np.mean(np.array(covs), axis=0)+alpha

    return calibrated_mean, calibrated_cov


def calibration_feture(support_data,support_label, base_means, base_data,n_shot, k=2,alpha=0.2):
    sampled_data = []
    sampled_label = []

    # support_data_mean = np.mean(support_data, axis=0, keepdims=True)
    # n_shot = 1
    num_sampled = int(512/n_shot)
    for i in range(n_shot):
        mean, cov = distribution_calibration(support_data[i], base_means, base_data,k=k,alpha=alpha)
        if(mean is None):
            sampled_data.append([support_data[i] for c in range(num_sampled)])
            sampled_label.extend([support_label[i]]*num_sampled)
            continue
        sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
        sampled_label.extend([support_label[i]]*num_sampled)
    sampled_data = np.concatenate([sampled_data[:]]).reshape(n_shot * num_sampled, -1)
    
    X_aug = np.concatenate([support_data, sampled_data])
    Y_aug = np.concatenate([support_label, sampled_label])
    return X_aug,Y_aug

def get_base_feature(feature):
# ---- Base class statistics
    base_means = []
    base_features_path = f"/root/datasets/FSCIL/base_feature/{feature}.pkl"
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            base_means.append(mean)
    return base_means,data