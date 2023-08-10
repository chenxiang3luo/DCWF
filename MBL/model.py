import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,kernel_size = 8,stride=1,act = 0):
        super(BasicBlock,self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.bias = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size,stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size,stride=stride)
        self.bn2 = nn.BatchNorm1d(planes)

        if act == 0:
            self.elu = nn.ELU(inplace=True)
        else:
            self.elu = nn.ReLU(inplace=True)
        
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
            BasicBlock(inplanes = 1,planes = 32,kernel_size = 8,stride = 1,act = 0),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0), 
            nn.Dropout(p=0.1),

            BasicBlock(inplanes = 32,planes = 64,kernel_size = 8,stride = 1,act=1),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0), 
            nn.Dropout(p=0.1),

            BasicBlock(inplanes = 64,planes = 128,kernel_size = 8,stride = 1,act=1),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0), 
            nn.Dropout(p=0.1),

            BasicBlock(inplanes = 128,planes = 256,kernel_size = 8,stride = 1,act=1),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0), 
            nn.Dropout(p=0.1),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),

        )

    def forward(self,input):
        x = self.features(input)
        x = self.classifier(x)
        return x

    def feature_forward(self,input):
        x = self.features(input)
        return x

