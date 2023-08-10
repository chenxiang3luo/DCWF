import pickle
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch as t
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

def create_test_set_AWF_disjoint(features_model,dataset,shot):
    n_query = 70


    train_name = f'/root/datasets/FSCIL/{dataset}.npz'
    train_dataset = np.load(train_name,allow_pickle=True)
    train_data = train_dataset['data']
    train_labels = train_dataset['labels']
    train_labels = np.array([str(lab) for lab in train_labels])
    unique = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    num_base_class = len(unique)
    print(num_base_class)
    
    dic_train = {}
    dic_test = {}
    for i in range(num_base_class):
        inds_train = np.argwhere(train_labels==unique[i])

        samples_train = inds_train.reshape(-1)

        support = np.array(train_data[samples_train][:shot])
        support = support.reshape(support.shape[0],1,support.shape[1])

        query = np.array(train_data[samples_train][shot:shot+n_query])
        query = query.reshape(query.shape[0],1,query.shape[1])

        support = t.tensor(support,dtype=t.float32).cuda()
        query = t.tensor(query,dtype=t.float32).cuda()
        features_model.eval()
        with t.no_grad():
            dic_train[unique[i]] = np.array([np.array(features_model(support).cpu()).mean(axis=0)])
            dic_test[unique[i]] = np.array(features_model(query).cpu())


    return dic_train, dic_test


def create_test_set_AWF_common(features_model,dataset_train,dataset_test,shot):
    n_query = 70


    train_name = f'/root/datasets/FSCIL/{dataset_train}.npz'
    train_dataset = np.load(train_name,allow_pickle=True)
    train_data = train_dataset['data']
    train_labels = train_dataset['labels']
    print(train_data.shape)
    train_labels = np.array([str(lab) for lab in train_labels])
    unique = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    num_base_class = len(unique)
    print(num_base_class)
    
    X_train,X_test, y_train, y_test =train_test_split(train_data,train_labels,test_size=(shot)*num_base_class, random_state=42,shuffle=True,stratify=train_labels)
    support_data = X_test
    support_label = y_test

    test_name = f'/root/datasets/FSCIL/{dataset_test}.npz'
    test_dataset = np.load(test_name,allow_pickle=True)
    test_data = test_dataset['data']
    test_labels = test_dataset['labels']
    test_labels = np.array([str(lab) for lab in test_labels])

    X_train,X_test, y_train, y_test =train_test_split(test_data,test_labels,test_size=(n_query)*num_base_class, random_state=42,shuffle=True,stratify=test_labels)
    query_data = X_test
    query_label = y_test

    dic_train = {}
    dic_test = {}
    for i in range(num_base_class):
        inds_train = np.argwhere(support_label==unique[i])
        inds_test = np.argwhere(query_label==unique[i])
        samples_train = inds_train.reshape(-1)
        samples_test = inds_test.reshape(-1)

        support = np.array(support_data[samples_train])
        support = support.reshape(support.shape[0],1,support.shape[1])

        query = np.array(query_data[samples_test])
        query = query.reshape(query.shape[0],1,query.shape[1])

        support = t.tensor(support,dtype=t.float32).cuda()
        query = t.tensor(query,dtype=t.float32).cuda()
        features_model.eval()
        with t.no_grad():
            dic_train[unique[i]] = np.array([np.array(features_model(support).cpu()).mean(axis=0)])
            dic_test[unique[i]] = np.array(features_model(query).cpu())


    return dic_train, dic_test



def create_test_set_AWF_training_included(features_model,dataset,shot,training_included):
    num_base_class = 100
    n_query = 70


    train_name = f'/root/datasets/FSCIL/AWF_775_TrainingIncluded.npz'
    train_dataset = np.load(train_name,allow_pickle=True)
    train_data = train_dataset['data']
    train_labels = train_dataset['labels']
    # if(shot != 20):
    #     X_train,X_test, y_train, y_test =train_test_split(train_data,train_labels,test_size=(shot+n_query)*775, random_state=42,shuffle=True,stratify=train_labels)
    # else:
    #     X_test = train_data
    #     y_test = train_labels
    train_labels = np.array([str(lab) for lab in train_labels])
    unique = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    # unique = unique[np.random.permutation(unique.shape[0])]
    trained_classes = unique[:training_included]

    dic_train = {}
    dic_test = {}
    for cla in trained_classes:
        inds_train = np.argwhere(train_labels==cla)

        samples_train = inds_train.reshape(-1)

        support = np.array(train_data[samples_train][:shot])
        support = support.reshape(support.shape[0],1,support.shape[1])

        query = np.array(train_data[samples_train][shot:shot+n_query])
        query = query.reshape(query.shape[0],1,query.shape[1])

        support = t.tensor(support,dtype=t.float32).cuda()
        query = t.tensor(query,dtype=t.float32).cuda()
        features_model.eval()
        with t.no_grad():
            dic_train[cla] = np.array([np.array(features_model(support).cpu()).mean(axis=0)])
            dic_test[cla] = np.array(features_model(query).cpu())

    train_name = f'/root/datasets/FSCIL/{dataset}.npz'
    train_dataset = np.load(train_name,allow_pickle=True)
    train_data = train_dataset['data']
    train_labels = train_dataset['labels']

    # X_train,X_test, y_train, y_test =train_test_split(train_data,train_labels,test_size=(shot+n_query)*num_base_class, random_state=42,shuffle=True,stratify=train_labels)
    train_labels = np.array([str(lab) for lab in train_labels])
    unique = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    # unique = unique[np.random.permutation(unique.shape[0])]
    test_classes = unique[:100-training_included]
    
    for cla in test_classes:
        inds_train = np.argwhere(train_labels==cla)

        samples_train = inds_train.reshape(-1)

        support = np.array(train_data[samples_train][:shot])
        support = support.reshape(support.shape[0],1,support.shape[1])

        query = np.array(train_data[samples_train][shot:shot+n_query])
        query = query.reshape(query.shape[0],1,query.shape[1])

        support = t.tensor(support,dtype=t.float32).cuda()
        query = t.tensor(query,dtype=t.float32).cuda()
        features_model.eval()
        with t.no_grad():
            dic_train[cla] = np.array([np.array(features_model(support).cpu()).mean(axis=0)])
            dic_test[cla] = np.array(features_model(query).cpu())


    return dic_train, dic_test



def kNN_accuracy(signature_vector_dict, test_vector_dict, dataset,n_shot):
    X_train = []
    y_train = []

    # print "Size of problem :", size_of_problem
    site_labels = list(signature_vector_dict.keys())
    print(len(site_labels))
    random.shuffle(site_labels)
    tested_sites = site_labels[:]
    for s in tested_sites:
        for each_test in range(len(signature_vector_dict[s])):
            X_train.append(signature_vector_dict[s][each_test])
            y_train.append(s)

    X_test = []
    y_test = []
    for s in tested_sites:
        for i in range(len(test_vector_dict[s])):
            X_test.append(test_vector_dict[s][i])
            y_test.append(s)

    knn = KNeighborsClassifier(n_neighbors=n_shot, weights='distance', p=2, metric='cosine', algorithm='brute')
    knn.fit(np.array(X_train),np.array(y_train))
    joblib.dump(knn, f'./trained_model/knn_model_{dataset}_{n_shot}.pkl')

    predict = knn.predict(X_test)
    acc_knn_top1 = accuracy_score(y_test,predict)
    tpr,fpr,f1 = get_matrix(predict,y_test)
    acc_knn_top1 = float("{0:.15f}".format(round(acc_knn_top1, 6)))


    return acc_knn_top1,tpr.mean(),fpr.mean(),f1.mean()


def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    
    for i in range(len(base_means)):
        # dist.append(np.linalg.norm(query-base_means[i]))
        dot_product = np.dot(query, base_means[i])
        # 计算向量的范数
        norm1 = np.linalg.norm(query)
        norm2 = np.linalg.norm(base_means[i])
        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)
        dist.append(1-similarity)
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov

def calibration_feture(support_data, base_means, base_cov,n_shot, k=2,alpha=0.21):
    sampled_data = []
    num_sampled = int(20/n_shot)
    for i in range(n_shot):
        mean, cov = distribution_calibration(support_data[i], base_means, base_cov,k=k,alpha=alpha)
        sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))

    sampled_data = np.concatenate([sampled_data[:]]).reshape(n_shot * num_sampled, -1)
    X_aug = np.concatenate([support_data, sampled_data])
    return X_aug

def get_base_feature(feature):
# ---- Base class statistics
    base_means = []
    base_cov = []
    base_features_path = f"./{feature}.pkl"
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)
    return base_means,base_cov


def create_test_set_AWF_open(features_model,dataset,shot,size):
    num_base_class = 100
    n_query = 70


    train_name = f'/root/datasets/FSCIL/{dataset}.npz'
    train_dataset = np.load(train_name,allow_pickle=True)
    train_data = train_dataset['data']
    train_labels = train_dataset['labels']
    X_train,X_test, y_train, y_test =train_test_split(train_data,train_labels,test_size=(shot+n_query)*100, random_state=42,shuffle=True,stratify=train_labels)
    y_test = np.array([lab for lab in y_test])
    unique = np.unique(y_test)
    unique = unique[np.random.permutation(unique.shape[0])]

    dic_train = {}
    dic_test = {}
    for cla in unique:
        inds_train = np.argwhere(y_test==cla)

        samples_train = inds_train.reshape(-1)

        support = np.array(X_test[samples_train][:shot])
        support = support.reshape(support.shape[0],1,support.shape[1])

        query = np.array(X_test[samples_train][shot:])
        query = query.reshape(query.shape[0],1,query.shape[1])

        support = t.tensor(support,dtype=t.float32).cuda()
        query = t.tensor(query,dtype=t.float32).cuda()
        features_model.eval()
        with t.no_grad():
            dic_train[cla] = np.array([np.array(features_model(support).cpu()).mean(axis=0)])
            dic_test[cla] = np.array(features_model(query).cpu())

    train_name = f'/root/datasets/FSCIL/tor_open_400000w.npz'
    train_dataset = np.load(train_name,allow_pickle=True)
    train_data = train_dataset['data']
    train_data = train_data[np.random.permutation(train_data.shape[0])]

    support = train_data[:shot]
    support = support.reshape(support.shape[0],1,support.shape[1])
    support = t.tensor(support,dtype=t.float32).cuda()
    features_model.eval()
    with t.no_grad():
        dic_train['opend_world'] = np.array([np.array(features_model(support).cpu()).mean(axis=0)])
    cc = []

    for i in range(shot,size,200):

        query = np.array(train_data[i:i+200])
        query = query.reshape(query.shape[0],1,query.shape[1])
        query = t.tensor(query,dtype=t.float32).cuda()
        features_model.eval()
        with t.no_grad():
            cc.extend(features_model(query).cpu().numpy())
    dic_test['opend_world'] = np.array(cc)

    return dic_train, dic_test