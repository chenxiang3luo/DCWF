#This turns dataset into [num, length] 1D sequences
import numpy as np
import math 
import os 
from os.path import join
import argparse
import logging
import configparser
from model.model_keras import *
import multiprocessing as mp
import h5py
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from tqdm import tqdm
logger = logging.getLogger('df')

def get_pkt_list(trace_data):
    first_line = trace_data[0]
    first_line = first_line.split()
    first_time = float(first_line[0])
    dta = []
    for line in trace_data:
        a = line
        b = a.split()

        if float(b[1]) > 0:
            #dta.append(((float(b[0])- first_time), abs(int(b[2])), 1))
            dta.append(((float(b[0])- first_time), 1))
        else:
            #dta.append(((float(b[1]) - first_time), abs(int(b[2])), -1))
            dta.append(((float(b[0]) - first_time), -1))
    return dta
def In_Out(list_data):
    In = []
    Out = []
    for p in list_data:
        if p[1] == -1:
            In.append(p)
        if p[1] == 1:
            Out.append(p)
    return In, Out

def perc_inc_out(trace_data):
    Total = get_pkt_list(trace_data)
    In, Out = In_Out(Total)
    percentage_in = len(In)/float(len(Total))
    percentage_out = len(Out)/float(len(Total))
    return percentage_in, percentage_out


def number_pkt_stats(trace_data):
    Total = get_pkt_list(trace_data)
    In, Out = In_Out(Total)
    return len(In), len(Out), len(Total)

def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)  
    return dict(cf['default'])

def parallel(flist,model): 
    data_dict = []
    for file in tqdm(flist):
        data_dict.append(extractfeature(file,model))
    return data_dict

def extractfeature(f,model):
    global MON_SITE_NUM
    fname = f.split('\\')[-1]
    # logger.info('Processing %s...'%fname)
#    try:
#        with open(f,'r') as f:
#            tcp_dump = f.readlines()
##            return tcp_dump, 1
#
#        feature = pd.Series(tcp_dump).str.slice(0,-2).str.split('\t',expand = True).astype(int)
#        print(feature)
#           feature = np.array(feature.iloc[:,1])
#
#        
#    except:
#        raise ValueError("..")
    with open(f,'r') as f:
        tcp_dump = f.readlines()
#            return tcp_dump, 1
    number_pkts = number_pkt_stats(tcp_dump)
    ratio = perc_inc_out(tcp_dump)
    feature = pd.Series(tcp_dump).str.slice(0,-1).str.split('\t',expand = True).astype("float")
    feature = np.array(feature.iloc[:,1]).astype("int")
    feature = pad_sequences([feature], padding ='post', truncating = 'post', value = 0, maxlen = 5000)
    feature = list(model.predict(feature).reshape(500))
    feature.extend(number_pkts)
    feature.extend(ratio)
    if '-' in fname:
        label = fname.split('-')
        label = int(label[0])
    else:
        label = MON_SITE_NUM
    return (feature,label)


if __name__== '__main__':
    global MON_SITE_NUM
    '''initialize logger'''
    MON_SITE_NUM = 100
    MON_INST_NUM = 100
    num_class = MON_SITE_NUM
    '''read in arg'''
    parser = argparse.ArgumentParser(description='DF feature extraction')
    parser.add_argument('-traces_path',
                        metavar='<traces path>',
                        help='Path to the directory with the traffic traces to be simulated.',
                        default='D:\\Work\\work\\TangwenYi\\Lab\\WebsiteFingerprinting-master\\attacks\\results\\20000')
    parser.add_argument('-format',
                        metavar='<trace suffix>',
                        default = '.cell',
                        help='trace suffix')
    parser.add_argument('-l',
                        metavar='<feature length>',
                        type = int,
                        default = 5000,
                        help='generated feature length')

    args = parser.parse_args()
    
  
    # fpath = os.path.join(args.traces_path, '*/*')
    # flist = glob.glob(fpath)
    # if len(flist) == 0:
    #     fpath = os.path.join(args.traces_path,'*')
    #     flist = glob.glob(fpath)
    #     if len(flist)  == 0:
    #         logger.error('Path {} is incorrect!'.format(fpath))


    # raw_data_dict = parallel(flist,n_jobs = 20)
    # features, label = zip(*raw_data_dict)
    # features = pad_sequences(features, padding ='post', truncating = 'post', value = 0, maxlen = LENGTH)
    
    # labels = to_categorical(label, num_classes = MON_SITE_NUM+1) 
    

    
    fpath = os.path.join(args.traces_path,'*')
    flist = []
    INPUT_SHAPE = (5000,1)
    model = DFNet.build_DAE(input_shape=INPUT_SHAPE)
    model.load_weights('./dae_weights')
    model = Model(inputs=model.input,outputs=model.get_layer('get_feature').output)
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            flist.append(join(args.traces_path, str(i)+'-'+str(j)+args.format))
    # for i in range(UNMON_SITE_NUM):
    #     flist.append(join(args.traces_path, str(i)+args.format))
    data_dict = {'feature':[],'label':[]}
    raw_data_dict = parallel(flist,model)
    features, label = zip(*raw_data_dict)
    # features = pad_sequences(features, padding ='post', truncating = 'post', value = 0, maxlen = args.l)
    features = np.array(features)
    labels = to_categorical(label, num_classes = num_class) 

    
    
    print("feature shape:{}, label shape:{}".format(features.shape, labels.shape))
    print(features.shape)
    data_dict['feature'], data_dict['label'] = features, labels
    np.save('./feature.npy',data_dict)        

