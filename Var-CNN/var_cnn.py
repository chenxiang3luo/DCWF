import os
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
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
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    kernel_size = 3
    pad = (kernel_size - 1) * dilation
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=pad, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        
        super(ResNet18, self).__init__()
        self.ini = nn.Sequential(
            nn.ConstantPad1d(3,0),
            nn.Conv1d(1,64,kernel_size=7,bias=False),
            nn.BatchNorm1d(64,eps=1e-5),
            nn.ReLU(),
            nn.MaxPool1d(3,2,padding=1)
        )

        self.inplanes = 64

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stage=0,block=0,dilation=[1,2]),
            BasicBlock(64, 64, stage=0,block=1,dilation=[4,8])
        )
        # 对应于 conv3
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stage=1,block=0,dilation=[1,2]),
            BasicBlock(128, 128, stage=1,block=1,dilation=[4,8])
        )
        # 对应于 conv4
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stage=2,block=0,dilation=[1,2]),
            BasicBlock(256, 256, stage=2,block=1,dilation=[4,8])
        )
        # 对应于 conv5
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stage=3,block=0,dilation=[1,2]),
            BasicBlock(512, 512, stage=3,block=1,dilation=[4,8])
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(512 , num_classes)


    def forward(self, x):
        # See note [TorchScript super()]

        # conv1
        # x: [3, 224, 224] -> [64, 112, 112]
        x = self.ini(x)
  # x: [64, 56, 56] -> [64, 56, 56]
  # x 经过第一个 layer, 大小是不变的
        x = self.layer1(x)

        # conv3
        # x: [64, 56, 56] -> [128, 28, 28]
        x = self.layer2(x)

        # conv4
        # x: [128, 28, 28] -> [256, 14, 14]
        x = self.layer3(x)

        # conv5
        # x: [256, 14, 14] -> [512, 7, 7]
        x = self.layer4(x)

  # x: [512, 7, 7] -> [512, 1, 1]
        x = self.avgpool(x)
        x = self.fc(x)

        return x
    def feature_forward(self, x):
        x = self.ini(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stage=0, block=0,stride=1,dilation=None):
        super(BasicBlock, self).__init__()
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2
        self.block = block
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = CausalConv1d(inplanes, planes, kernel_size=3,stride=stride,dilation=dilation[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = CausalConv1d(planes, planes, kernel_size=3,stride=1,dilation=dilation[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.ini = nn.Sequential(
            conv1x1(inplanes,planes,stride=stride),
            nn.BatchNorm1d(planes)
        )
    def forward(self, x):
        identity = x
  # 如果是 layer2，layer3，layer4 里的第一个 BasicBlock，第一个卷积层会降采样
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.block == 0:
            identity = self.ini(x)

        out += identity
        out = self.relu(out)

        return out
