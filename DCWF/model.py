#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import numpy as np
from sklearn import svm, datasets
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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''

        torch.save(model.state_dict(), './finish_model_open.pt')                 # 这里会存储迄今最优的模型
        print('save finish_model_open')

class EarlyStopping_size:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model,size):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,size)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,size):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        torch.save(model.state_dict(), './finish_model_front_{}.pt'.format(size))                 # 这里会存储迄今最优的模型
        print('save finish_model_{}'.format(size))

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, ratio=2):
        super().__init__()
 
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.network = nn.Sequential(
            nn.Linear(channel, channel//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//ratio, channel,  bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, inputs):
        b, c, _ = inputs.shape
        x = torch.var(inputs,dim=-1)
        # x = self.avg_pool(inputs)
        x = x.view(b, c)
        x = self.network(x)
        x = x.view(b, c, 1)
        x = inputs * x
        return x
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))
# import warnings
# import torch.nn as nn
class Bottleneck(nn.Module):
    
    
    def __init__(self, stage, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 2
        self.conv1 = nn.Conv1d(stage, stage, kernel_size=1,padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(stage)
        self.conv2 = nn.Conv1d(stage, stage, kernel_size=31, padding='same',stride=stride, bias=False,groups=stage)
        self.bn2 = nn.BatchNorm1d(stage)
        
        self.conv3 = nn.Conv1d(stage, stage*self.expansion, kernel_size=1,padding='same', bias=False)
        self.bn3 = nn.BatchNorm1d(stage*self.expansion)
        # self.attention = SqueezeAndExcitation(stage*self.expansion)
        self.shortcut = nn.Sequential(
            nn.Conv1d(stage, stage*self.expansion, kernel_size=1,padding='same', stride=1, bias=False),
            nn.BatchNorm1d(stage*self.expansion)
        )

    def forward(self, x):
        out = nn.functional.elu(self.bn2(self.conv1(x)))
        out = nn.functional.elu(self.bn1(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))
        # out = self.attention(out)
        out += self.shortcut(x)
        out = nn.functional.elu(out)
        return out
# from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

class SeparableConv1D(nn.Module):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, input_filters, output_filters, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            padding=(kernel_size-1) // 2,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))

    def forward(self, hidden_states):
        x = self.depthwise(hidden_states)
        x = self.pointwise(x)
        x += self.bias
        return x
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.resconv = nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=1)

        self.conv1 = nn.Sequential(
            # PackedDeformConv1d(in_channels,out_channels,kernel_size=4,padding='same',bias=False,groups=1),
            nn.Conv1d(in_channels, out_channels,kernel_size=8,padding=3)
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Sequential(
            # PackedDeformConv1d(out_channels,out_channels,kernel_size=8,padding='same',bias=False,groups=2),
            nn.Conv1d(out_channels, out_channels,kernel_size=8,padding=4)
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.elu2 = nn.ELU()
        self.downsample = downsample
        
    def forward(self, x):
        residual = self.resconv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.bn2(out)
        out = self.elu2(out)
        return out
    
class DF(nn.Module):
    """
    for CIFAR10
    """
    def __init__(self,num_classes=120):
        super(DF, self).__init__()
        self.activation = []
        # 两层卷积 
        self.ini = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=31,padding='same',bias=False,groups=1),
            # nn.LayerNorm([32,5000]),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3))
        self.se_attention = nn.Sequential(
            SqueezeAndExcitation(32))
        self.conv_unit = nn.Sequential(
            
            Bottleneck(32),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),# 池化
            nn.Dropout(0.1),
                       
            Bottleneck(64),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),# 池化    
            nn.Dropout(0.1),
            
            Bottleneck(128),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),# 池化    
            nn.Dropout(0.1),

            Bottleneck(256),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),# 池化    
            # nn.AdaptiveAvgPool1d(1),
            nn.Flatten()

        )

        self.fc_unit = nn.Sequential(

            nn.Linear(512, num_classes),

        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
    def reset_attention(self):
        self.activation = []
    def hook_register(self):
        def hook(module, fea_in, fea_out):
            self.activation.append(fea_out.cpu().detach().numpy())

        for name,m in self.named_modules():
            if('network.3' in name):
                print('registering',name)
                m.register_forward_hook(hook=hook)

    def forward(self, x):# 数据从此进来，经过定义好的各层网络，最终输出

        x = self.ini(x)
        x = self.se_attention(x)
        x = self.conv_unit(x)
        
        # logits = self.fc_unit(x)

        return x
    def feature_forward(self,x):
        x = self.ini(x)
        x = self.se_attention(x)
        x = self.conv_unit(x)
        
        return x

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes,kernel_size = 8,stride=1,act = 0):
#         super(BasicBlock,self).__init__()
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         # self.bias = nn.Parameter(torch.ones(1))
#         self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size,stride=stride)
#         self.bn1 = nn.BatchNorm1d(planes)
#         if act == 0:
#             self.elu = nn.ELU(inplace=True)
#         else:
#             self.elu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size,stride=stride)
#         self.bn2 = nn.BatchNorm1d(planes)
        
#     def forward(self, x):
#   # 如果是 layer2，layer3，layer4 里的第一个 BasicBlock，第一个卷积层会降采样
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.elu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out = self.elu(out)

#         return out
# class DF(nn.Module):
#     def __init__(self,emd_size=256):
#         super(DF, self).__init__()
#         self.features = nn.Sequential(
#             BasicBlock(inplanes = 1,planes = 32,kernel_size = 8,stride = 1,act = 0),
#             nn.MaxPool1d(kernel_size=8, stride=4, padding=0),
#             nn.Dropout(p=0.1),
#             BasicBlock(inplanes = 32,planes = 64,kernel_size = 8,stride = 1,act=1),
#             nn.MaxPool1d(kernel_size=8, stride=4, padding=0), 
#             nn.Dropout(p=0.1),
#             BasicBlock(inplanes = 64,planes = 128,kernel_size = 8,stride = 1,act=1),
#             nn.MaxPool1d(kernel_size=8, stride=4, padding=0),  
#             nn.Dropout(p=0.1),
#             BasicBlock(inplanes = 128,planes = 256,kernel_size = 8,stride = 1,act=0),
#             nn.MaxPool1d(kernel_size=8, stride=4, padding=0),  
#             # nn.Dropout(p=0.1),
#             # nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             # nn.Linear(3328, emd_size),
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(3328, emd_size),
            

#         )

#     def forward(self,input):
#         x = self.features(input)
#         return x
#     def feature_forward(self,input):
#         x = self.features(input)
#         return x