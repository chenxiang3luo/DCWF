import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from util.BDC import *
import numpy as np
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
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,kernel_size = 8,stride=1,act = 0, deep = 1):
        super(BasicBlock,self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.bias = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size,stride=stride,padding = 1)
        shape = 5000 // deep
        # self.bn1 = nn.BatchNorm1d(planes)
        self.bn1 = nn.LayerNorm([planes, shape-1])

        if act == 0:
            self.elu = nn.ELU(inplace=True)
        else:
            self.elu = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size,stride=stride,padding = 2)
        # self.bn2 = nn.BatchNorm1d(planes)
        self.bn2 = nn.LayerNorm([planes,shape])
        
    def forward(self, x):
  # 如果是 layer2，layer3，layer4 里的第一个 BasicBlock，第一个卷积层会降采样
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.elu(out)

        return out
class DF(nn.Module):
    def __init__(self,num_classes=100):
        super(DF, self).__init__()
        self.features = nn.Sequential(
            BasicBlock(inplanes = 1,planes = 32,kernel_size = 4,stride = 1,act = 0,deep = 1),
            nn.MaxPool1d(kernel_size=4, stride=4, padding = 0),
            nn.Dropout(p=0.1),
            BasicBlock(inplanes = 32,planes = 64,kernel_size = 4,stride = 1,act = 1,deep = 4),
            nn.MaxPool1d(kernel_size=4, stride=4, padding = 0), 
            nn.Dropout(p=0.1),
            BasicBlock(inplanes = 64,planes = 128,kernel_size = 4,stride = 1,act = 1,deep = 16),
            nn.MaxPool1d(kernel_size=4, stride=4, padding = 0),  
            nn.Dropout(p=0.1),
            BasicBlock(inplanes = 128,planes = 256,kernel_size = 4,stride = 1,act = 1,deep = 64),
            nn.MaxPool1d(kernel_size=4, stride=4, padding = 0),  
            nn.Dropout(p=0.1)
        )
        

    def forward(self,input):
        x = self.features(input)
        x = BDCovpool(x,torch.zeros(1).cuda())
        x = Triuvec(x)
        return x
    def feature_forward(self,input):
        x = self.features(input)
        return x
