import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F




class DF(nn.Module):
    """
    for CIFAR10
    """
    def __init__(self,num_classes=120):
        super(DF, self).__init__()

        # 两层卷积 
        self.conv_unit = nn.Sequential(
            
            # 3表示input，可以理解为图片的通道数量，即我的卷积核一次要到几个tensor去作卷积
            # 6表示有多少个卷积核
            # stride表示卷积核移动步长，padding表示边缘扩充
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=0),# 卷积
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
                   
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0),# 卷积
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),# 池化
            nn.Dropout(p=0.1),
                       

            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),# 卷积
            nn.BatchNorm1d(64),
            nn.ReLU(),
                   
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=0),# 卷积
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),# 池化
            nn.Dropout(p=0.1),
                       
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0),# 卷积
            nn.BatchNorm1d(128),
            nn.ReLU(),
                   
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=0),# 卷积
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),# 池化    
            nn.Dropout(p=0.1),
            
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=0),# 卷积
            nn.BatchNorm1d(256),
            nn.ReLU(),
                   
            nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=0),# 卷积
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),# 池化    
            nn.Dropout(p=0.1),
            nn.Flatten()



            # nn.AdaptiveAvgPool1d(1),
            # nn.Flatten()

        )

        self.fc_unit = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.Linear(256,num_classes,bias=False)
            nn.Linear(13*256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0,7),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0,5),

            nn.Linear(512, num_classes),
            # nn.Softmax(dim=0)
        )
        # # 3层全连接层
        # self.fc_unit = nn.Sequential(
        #     nn.Linear(5120, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.7),

        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, num_classes)
        # )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()

    def forward(self, x):# 数据从此进来，经过定义好的各层网络，最终输出
        batchsz = x.size(0)
        x = self.conv_unit(x)
        logits = self.fc_unit(x)
        return logits
    def forward_conv(self,x):
        x = self.conv_unit(x)
        return x