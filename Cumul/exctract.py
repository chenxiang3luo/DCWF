import numpy as np
import sys
import os
from os import mkdir, listdir
from os.path import join, isdir, dirname
from time import strftime
import pandas as pd
# import constants as ct
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
import configparser
import argparse
import logging
import glob
import multiprocessing as mp
logger = logging.getLogger('cumul')
np.random.seed(7)
def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)  
    return dict(cf['default'])


# def parse_arguments():

#     parser = argparse.ArgumentParser(description='It simulates adaptive padding on a set of web traffic traces.')

#     parser.add_argument('-t',
#                         metavar='<traces path>',
#                         help='Path to the directory with the traffic traces to be simulated.',
#                         default='D:\\Work\\work\\TangwenYi\\Lab\\WebsiteFingerprinting-master\\attacks\\results\\20000')

#     parser.add_argument('--log',
#                         type=str,
#                         dest="log",
#                         metavar='<log path>',
#                         default='stdout',
#                         help='path to the log file. It will print to stdout by default.')


#     # Parse arguments
#     args = parser.parse_args()
#     config_logger(args)
#     return args


# def config_logger(args):
#     # Set file
#     log_file = sys.stdout
#     if args.log != 'stdout':
#         log_file = open(args.log, 'w')
#     ch = logging.StreamHandler(log_file)

#     # Set logging format
#     ch.setFormatter(logging.Formatter(ct.LOG_FORMAT))
#     logger.addHandler(ch)

#     # Set level format
#     logger.setLevel(logging.INFO)



# def parse(fpath):
#     '''Parse a file assuming Tao's format.'''
#     t = []
#     for line in open(fpath):
#         try:    
#             timestamp, length = line.strip().split(ct.TRACE_SEP)
#             t.append(-int(length))
#         except ValueError:
#             logger.warn("Could not split line: %s in %s", line, fpath)      
#             continue    
#     return t

def extract(sinste):
    #sinste: list of packet sizes

    #first 4 features

    insize = 0
    outsize = 0
    inpacket = 0
    outpacket = 0

    for i in range(0, len(sinste)):
        if sinste[i] > 0:
            outsize += sinste[i]
            outpacket += 1
        else:
            insize += abs(sinste[i])
            inpacket += 1
    features = [insize, outsize, inpacket, outpacket]

    #100 interpolants
    
    n = 100 #number of linear interpolants

    x = 0 #sum of packet sizes
    y = 0 #sum of absolute packet sizes
    graph = []
    
    for si in range(0, len(sinste)):
        x += abs(sinste[si])
        y += sinste[si]
        graph.append([x, y])

    #derive interpolants
    max_x = graph[-1][0] 
    gap = float(max_x)/n
    cur_x = 0
    cur_y = 0
    graph_ptr = 0

    for i in range(0, n):
        #take linear line between cur_x and cur_x + gap
        next_x = cur_x + gap
        while (graph[graph_ptr][0] < next_x):
            graph_ptr += 1
            if (graph_ptr >= len(graph) - 1):
                graph_ptr = len(graph) - 1
                #wouldn't be necessary if floats were floats
                break
##        print "graph_ptr=", graph_ptr
        next_pt_y = graph[graph_ptr][1] #not next_y 
        next_pt_x = graph[graph_ptr][0]
        cur_pt_y = graph[graph_ptr-1][1]
        cur_pt_x = graph[graph_ptr-1][0]
##        print "lines are", cur_pt_x, cur_pt_y, next_pt_x, next_pt_y

        if (next_pt_x - cur_pt_x != 0):
            slope = (next_pt_y - cur_pt_y)/(next_pt_x - cur_pt_x)
        else:
            slope = 1000
        next_y = slope * (next_x - cur_pt_x) + cur_pt_y

##        print "xy are", cur_x, cur_y, next_x, next_y, max_x
        interpolant = (next_y - cur_y)/(next_x - cur_x)
        # features.append(interpolant)
        features.append(next_y)
        cur_x = next_x
        cur_y = next_y

    return features
def parallel(flist,n_jobs=20): 
    data_dict = []
    pool = mp.Pool(n_jobs)
    data_dict =  pool.map(extractfeature, flist)
    return data_dict

def extractfeature(data):
    (dump,label) = data
    dump = np.delete(dump, np.where(dump == 0))
    if(len(dump)==0):
        return (np.zeros(104),label)
    dump = np.negative(dump)
    # logger.info('Processing %s...'%f)
    features = extract(dump)
    
    return (features,label)
class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self
def get_features(session,name):
	# parser config and arguments
    # args = parse_arguments()
    # logger.info("Arguments: %s" % (args))

    MON_SITE_NUM = 100
    # MON_INST_NUM = int(cf['monitored_inst_num'])

    # OPEN_WORLD = 0

    # logger.info('Extracting features...')
    session = session
    data_dict_train = {'feature':[],'label':[]}
    data_dict_test = {'feature':[],'label':[]}
    # base_class = 100
    # shot = 5
    # way = 20
    # query = 70
    # train_name = '/root/Lab/train_data_df.npz'
    # test_name = '/root/dataset/dataset/test_data_df.npz'
    # train_dataset_fw = np.load(train_name,allow_pickle=True)
    # test_dataset_fw = np.load(test_name,allow_pickle=True)
    # train_data = train_dataset_fw['data'][:(base_class+way*session)*shot]
    # train_labels = train_dataset_fw['labels'][:(base_class+way*session)*shot]
    # train_data = train_data.reshape(train_data.shape[0],1,train_data.shape[1])

    # test_data = test_dataset_fw['data'][:(base_class+way*session)*query]
    # test_labels = test_dataset_fw['labels'][:(base_class+way*session)*query]
    # test_data = test_data.reshape(test_data.shape[0],1,test_data.shape[1])
    # num_base_class = 100
    # shot = 5
    # query = 70
    # way = 20
    base_class = 50
    shot = 5
    way = 10
    query = 95
    train_name = '/root/Lab/train_data_fw_DF_front.npz'
    test_name = '/root/Lab/test_data_fw_DF_front.npz'
    train_dataset_fw = np.load(train_name,allow_pickle=True)
    test_dataset_fw = np.load(test_name,allow_pickle=True)
    train_data = train_dataset_fw['data'][:(base_class+way*session)*shot]
    train_labels = train_dataset_fw['labels'][:(base_class+way*session)*shot]
    train_data = train_data.reshape(train_data.shape[0],1,train_data.shape[1])

    test_data = test_dataset_fw['data'][:(base_class+way*session)*query]
    test_labels = test_dataset_fw['labels'][:(base_class+way*session)*query]
    test_data = test_data.reshape(test_data.shape[0],1,test_data.shape[1])
    # train_dataset_fw = np.load('/root/dataset/dataset/train_data_fw.npz',allow_pickle=True)
    # test_dataset_fw = np.load('/root/dataset/dataset/test_data_fw.npz',allow_pickle=True)

    # train_dataset_fw_DF = np.load('/root/Lab/train_data_fw_DF.npz',allow_pickle=True)
    # test_dataset_fw_DF = np.load('/root/Lab/test_data_fw_DF.npz',allow_pickle=True)
    # if(session==0):
    #     train_data = train_dataset_fw['data'][:num_base_class*shot]
    #     train_labels = train_dataset_fw['labels'][:num_base_class*shot]
    #     test_data = test_dataset_fw['data'][:(num_base_class)*query]
    #     test_labels = test_dataset_fw['labels'][:(num_base_class)*query]
    # else:
    #     train_data = train_dataset_fw['data'][:num_base_class*shot]
    #     train_labels = train_dataset_fw['labels'][:num_base_class*shot]
    #     test_data = test_dataset_fw['data'][:num_base_class*query]
    #     test_labels = test_dataset_fw['labels'][:num_base_class*query]

    #     train_data_DF = train_dataset_fw_DF['data'][:(way*session)*shot]
    #     train_labels_DF = train_dataset_fw_DF['labels'][:(way*session)*shot]+100
    #     test_data_DF = test_dataset_fw_DF['data'][:(way*session)*95]
    #     test_labels_DF = test_dataset_fw_DF['labels'][:(way*session)*95]+100
    #     train_data = np.concatenate([train_data,train_data_DF])
    #     train_labels = np.concatenate([train_labels,train_labels_DF])
    #     test_data = np.concatenate([test_data,test_data_DF])
    #     test_labels = np.concatenate([test_labels,test_labels_DF])
    # test_data = test_data.reshape(test_data.shape[0],1,test_data.shape[1])
    # train_data = train_data.reshape(train_data.shape[0],1,train_data.shape[1])
    data_test = zip(test_data,test_labels)
    data_train = zip(train_data,train_labels)
    raw_data_test = parallel(data_test)
    raw_data_train = parallel(data_train)
    data_dict_train['feature'], data_dict_train['label'] = zip(*raw_data_train)
    data_dict_test['feature'], data_dict_test['label'] = zip(*raw_data_test)
    outputdir = './feature_train.npy'
    np.save(outputdir,data_dict_train)
    outputdir = './feature_test.npy'
    np.save(outputdir,data_dict_test) 
    # for f in flist:
    #     if os.path.exists(f):
    #         fname = f.split('/')[-1]
    #         logger.info('Processing %s...'%f)
    #         data_dict['feature'].append(extract(parse(f)))
    #         if '-' in fname:
    #             label = int(fname.split('-')[0])
    #         else:
    #             label = int(MON_SITE_NUM)
            
    #         data_dict['label'].append(label)
    #     else:
    #         logger.warn('File %s does not exist!'%f)
##    
##    for i in range(MON_SITE_NUM):
##        for j in range(MON_INST_NUM):
##            fname = str(i)+'-'+str(j) 
##            if os.path.exists(join(args.t,fname)):
##                logger.info('Processing %s...'%fname)
##                data_dict['feature'].append(extract(parse(join(args.t, fname))))
##                data_dict['label'].append(i)
##            else:
##                logger.warn('File %s does not exist!'%fname)
##
##    if OPEN_WORLD:
##        for i in range(UNMON_SITE_NUM):
##            if os.path.exists(join(args.t,str(i))):
##                logger.info('Processing %s...'%i)
##                data_dict['feature'].append(extract(parse(join(args.t, str(i)))))
##                data_dict['label'].append(MON_SITE_NUM)
##            else:
##                logger.warn('File %s does not exist!'%i)
#
    