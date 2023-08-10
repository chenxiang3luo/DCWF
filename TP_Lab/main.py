import torch as t
from DF_model_Torch import DF
from utility import *
import os
from tqdm import tqdm
import torch
import numpy as np
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = False  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


def AWF_Disjoint_Experment(dataset):
    '''
    This function aims to experiment the performance of TF attack
    when the model is trained on the dataset with similar distribution.
    The model is trained on AWF777 and tested on AWF100 and the set of websites
    in the training set and the testing set are mutually exclusive.
    '''
    model_path = './trained_model/AWF_775_model.pt'
    checkpoint = t.load(model_path)
    feature_model = DF(64)
    feature_model.cuda()
    feature_model.load_state_dict(checkpoint)

    type_exp = 'N-MEV'
    # N-MEV is the use of mean of embedded vectors as mentioned in the paper
    SOP_list = [100]
    n_shot_list = [1,5,10,15,20]

    for size_of_problem in SOP_list:
        print("SOP:", size_of_problem)
        for n_shot in n_shot_list:
            acc_list_Top1 = []
            signature_vector_dict, test_vector_dict = create_test_set_AWF_disjoint(features_model=feature_model,
                                                                                   dataset=dataset,shot = n_shot)
            # signature_vector_dict, test_vector_dict = create_test_set_AWF_open(features_model=feature_model,
            #                                                                         dataset='tor_100w_2500tr',shot = n_shot,size = 9000)
            # Measure the performance (accuracy)
            acc_knn_top1,tpr,fpr,f1 = kNN_accuracy(signature_vector_dict, test_vector_dict,dataset,n_shot=n_shot)
            acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
            print("N_shot:", n_shot)
            print(str(acc_list_Top1).strip('[]'),tpr,fpr,f1)



def AWF_TrainingIncluded_Experment():
    '''
    This function aims to experiment the performance of TF attack
    when the model is trained on the dataset with similar distribution.
    The model is trained on AWF777 and tested on AWF100 and the set of websites
    in the training set and the testing set are mutually exclusive.
    '''
    model_path = './trained_model/AWF_775_model.pt'
    checkpoint = t.load(model_path)
    feature_model = DF(64)
    feature_model.cuda()
    feature_model.load_state_dict(checkpoint)
    # for layer in features_model.layers:
    #     if 'residual' not in layer.name:
    #         layer.trainable = False
    type_exp = 'N-MEV'
    # N-MEV is the use of mean of embedded vectors as mentioned in the paper
    SOP_list = [100]
    # SOP_list is the size of problem (how large the closed world is)
    # You can run gird search for various sizes of problems
    # SOP_list = [100, 75, 50, 25, 10]
    n_shot_list = [1,5,10,15,20]
    # n_shot_list is the number of n examples (n-shot)
    # You can run grid search for various sizes of n-shot
    # n_shot_list = [1, 5, 10, 15, 20]
    for include in [0,25,50,75,100]:
        for size_of_problem in SOP_list:
            print("SOP:", size_of_problem)
            for n_shot in n_shot_list:
                acc_list_Top1 = []
                signature_vector_dict, test_vector_dict = create_test_set_AWF_training_included(features_model=feature_model,
                                                                                    dataset='tor_100w_2500tr',shot = n_shot,training_included=include)

                # Measure the performance (accuracy)
                acc_knn_top1 = kNN_accuracy(signature_vector_dict, test_vector_dict, n_shot=n_shot)
                acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
                print("N_shot:", n_shot)
                print(str(acc_list_Top1).strip('[]'))

setup_seed(42)
for dataset in ['tor_100w_2500tr','KNN','DF19','DF95']:
    print(dataset)
    AWF_Disjoint_Experment(dataset)
