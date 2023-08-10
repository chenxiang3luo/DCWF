#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import numpy as np
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
            self.save_checkpoint(val_loss, model)
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
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, './finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss


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
        x = self.avg_pool(inputs)
        x = x.view(b, c)
        x = self.network(x)
        x = x.view(b, c, 1)
        x = inputs * x
        return x
 
import warnings
import torch.nn as nn

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

class SeparableConv1D(nn.Module):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, input_filters, output_filters, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            padding=kernel_size // 2,
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
        self.conv1 = SeparableConv1D(in_channels, out_channels,kernel_size=5)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu1 = nn.ELU()
        self.conv2 = SeparableConv1D(out_channels, out_channels,kernel_size=5)
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
    
class ResidualBlock_728(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock_728, self).__init__()
        self.resconv = nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=1)
        self.conv1 = SeparableConv1D(in_channels, in_channels,kernel_size=5)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = SeparableConv1D(in_channels, out_channels,kernel_size=5)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = self.resconv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.bn2(out)
        out = self.relu(out)
        return out

class DF(nn.Module):
    """
    for CIFAR10
    """
    def __init__(self,num_classes=100):
        super(DF, self).__init__()

        # 两层卷积
        self.conv_unit1 =  nn.Sequential(
            
            # 3表示input，可以理解为图片的通道数量，即我的卷积核一次要到几个tensor去作卷积
            # 6表示有多少个卷积核
            # stride表示卷积核移动步长，padding表示边缘扩充
            nn.Conv1d(1, 32, kernel_size=5, stride=2,padding=2),# 卷积
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.Conv1d(32, 64, kernel_size=5, stride=2,padding=2),# 卷积
            nn.BatchNorm1d(64),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=5, stride=2),# 池化
            )
        self.conv_unit2 =  nn.Sequential(
            SqueezeAndExcitation(64),
            ResidualBlock(64,128),
            ResidualBlock(128,256),
            ResidualBlock(256,728), 
            ResidualBlock_728(728,1024)
            )
        self.conv_unit3 =  nn.Sequential(
            SeparableConv1D(1024,1536,kernel_size=5),
            nn.BatchNorm1d(1536),
            nn.ELU(),
            SeparableConv1D(1536,2048,kernel_size=5),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            SqueezeAndExcitation(2048),
            nn.AdaptiveAvgPool1d(1),
            )

        self.fc_unit = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Softmax()
        )


    def forward(self, x):# 数据从此进来，经过定义好的各层网络，最终输出
        
        x = self.conv_unit1(x)
        x = self.conv_unit2(x)
        x = self.conv_unit3(x)
        b, c, _ = x.shape
        x= x.view(b,c)
        logits = self.fc_unit(x)
        return logits
    

class DAE(nn.Module):
    """
    for CIFAR10
    """
    def __init__(self):
        super(DAE, self).__init__()

        # 两层卷积
        self.encoder =  nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(5000,750),
            nn.ReLU(),
        )
        self.hider =  nn.Sequential(
            nn.Linear(750,500),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(500,750),
            nn.ReLU(),
            nn.Linear(750,5000)
        )

    def forward(self, x):# 数据从此进来，经过定义好的各层网络，最终输出
        
        x = self.encoder(x)
        x = self.hider(x)
        x = self.decoder(x)
        return x
    def forward_feature(self, x):# 数据从此进来，经过定义好的各层网络，最终输出
        
        x = self.encoder(x)
        x = self.hider(x)
        return x