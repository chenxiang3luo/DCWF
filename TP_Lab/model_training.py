import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from DF_model_Torch import DF,triplet_loss
import os
import random
import numpy as np
from tqdm import tqdm
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self



alpha = 0.1
batch_size_value = 128
emb_size = 64
number_epoch = 30
num_base_class = 100
num_sample = 25
# description = 'Triplet_Model'
# Training_Data_PATH = './dataset/extracted_AWF775/' 
# print(Training_Data_PATH)
print("with parameters, Alpha: %s, Batch_size: %s, Embedded_size: %s, Epoch_num: %s"%(alpha, batch_size_value, emb_size, number_epoch))

SEED=42
np.random.seed(SEED)
t.manual_seed(SEED)
t.cuda.manual_seed_all(SEED)
alpha_value = float(alpha)
# print(description)

train_dataset_500 = np.load('/root/datasets/FSCIL/KNN_WTFPAD.npz',allow_pickle=True)
base_data = train_dataset_500['data']
y_labels = train_dataset_500['labels']
X_train,X_test, y_train, y_test =train_test_split(base_data,y_labels,test_size=num_sample*num_base_class, random_state=42,shuffle=True,stratify=y_labels)
y_labels = y_test
base_data = np.array(X_test)
uni_labels = np.unique(y_labels)
id_to_classid = {k: v for k, v in enumerate(y_labels)}
classid_to_ids = {k: np.argwhere(y_labels == k).flatten() for k in np.unique(y_labels)}
base_data = base_data.reshape(base_data.shape[0],1,base_data.shape[1])
num_classes  = num_base_class


all_traces = base_data

print("Load traces with ",all_traces.shape)

def setup_seed(seed):
     t.manual_seed(seed)
     t.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def build_pos_pairs_for_id(classid): # classid --> e.g. 0
    traces = classid_to_ids[classid]

    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs

def build_positive_pairs(class_id_range):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    # random.permutation([1,2,3]) --> [2,1,3] just random
    # random.permutation(5) --> [1,0,4,3,2]
    # In this case, we just create the random index
    # Then return pairs of (identity, positive example)
    # that each element in pairs in term of its index is randomly ordered.
    return np.array(listX1)[perm], np.array(listX2)[perm]

Xa_train, Xp_train = build_positive_pairs(uni_labels)

# Gather the ids of all network traces that are used for training
# This just union of two sets set(A) | set(B)
all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
print("X_train Anchor: ", Xa_train.shape)
print("X_train Positive: ", Xp_train.shape)

# Build a loss which doesn't take into account the y_true, as# Build
# we'll be passing only 0
# def identity_loss(y_true, y_pred):
#     return K.mean(y_pred - 0 * y_true)

# The real loss is here
# def cosine_triplet_loss(X):
#     _alpha = alpha_value
#     positive_sim, negative_sim = X

#     losses = K.maximum(0.0, negative_sim - positive_sim + _alpha)
#     return K.mean(losses)

# ------------------- Hard Triplet Mining -----------
# Naive way to compute all similarities between all network traces.

def build_similarities(conv, all_imgs):
    batch_size = 900
    all_imgs = t.tensor(all_imgs,dtype=t.float32).cuda()
    conv.eval()
    num_imgs = all_imgs.size(0)
    with t.no_grad():
        embs = []
        for i in range(0, num_imgs, batch_size):
            batch_imgs = all_imgs[i:i+batch_size]
            batch_embs = conv(batch_imgs)
            embs.append(batch_embs)
    embs = t.cat(embs)
    embs = embs.cpu() / np.linalg.norm(embs.cpu(), axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    return all_sims

def intersect(a, b):
    return list(set(a) & set(b))

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
    # If no similarities were computed, return a random negative
    if similarities is None:
        return random.sample(neg_imgs_idx,len(anc_idxs))
    final_neg = []
    # for each positive pair
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        #positive similarity
        sim = similarities[anc_idx, pos_idx]
        # find all negatives which are semi(hard)
        possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg

class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, conv):
        self.batch_size = batch_size

        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.all_anchors = list(set(Xa_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.neg_traces_idx)}
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0

            # fill one batch
            traces_a = self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)

            yield self.traces[traces_a],self.traces[traces_p],self.traces[traces_n]
                   


batch_size = batch_size_value
setup_seed(42)

feature_model = DF(emb_size=emb_size)
optimizer = t.optim.SGD(feature_model.parameters(),lr=0.001,weight_decay=1e-6,momentum=0.9,nesterov=True)
cos_sim = nn.CosineSimilarity()
criterion = triplet_loss(cos_sim,alpha)
t.cuda.set_device(0)
feature_model.cuda(0)
criterion = criterion.cuda(0)
gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None).next_train()
# At first epoch we don't generate hard triplets
nb_epochs = number_epoch
for epoch in tqdm(range(nb_epochs),desc='Epoch'):
    for i in range(Xa_train.shape[0] // batch_size+1): 
        t.cuda.empty_cache()
        anchor,positive,negative = next(gen_hard)
        anchor = t.tensor(anchor,dtype=t.float32).cuda()
        positive = t.tensor(positive,dtype=t.float32).cuda()
        negative = t.tensor(negative,dtype=t.float32).cuda()
        feature_model.train()
        optimizer.zero_grad() 
        anchor_feature = feature_model(anchor) 
        positive_feature = feature_model(positive) 
        negative_feature = feature_model(negative) 

        loss = criterion(anchor_feature,positive_feature,negative_feature)
        loss.backward()
        optimizer.step()
    print(loss)
    t.save(feature_model.state_dict(), './trained_model/KNN_WTFPAD.pt')
    gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, feature_model).next_train()
t.save(feature_model.state_dict(), './trained_model/KNN_WTFPAD.pt')