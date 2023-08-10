import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
class DF(nn.Module):
    """
    for CIFAR10
    """
    def __init__(self,emb_size=128):
        super(DF, self).__init__()

        # 两层卷积
        self.conv_unit = nn.Sequential(
            
            # 3表示input，可以理解为图片的通道数量，即我的卷积核一次要到几个tensor去作卷积
            # 6表示有多少个卷积核
            # stride表示卷积核移动步长，padding表示边缘扩充
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding='same'),# 卷积
            nn.ELU(alpha=1.0),
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding='same'),# 卷积
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),# 池化
            nn.Dropout(p=0.1),
                       

            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding='same'),# 卷积
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding='same'),# 卷积
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),# 池化
            nn.Dropout(p=0.1),
                       
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding='same'),# 卷积
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding='same'),# 卷积
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),# 池化    
            nn.Dropout(p=0.1),
            
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding='same'),# 卷积
            nn.ReLU(), 
            nn.Conv1d(256, 256, kernel_size=8, stride=1, padding='same'),# 卷积
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),# 池化    

            nn.Flatten(),

            nn.Linear(256*20, emb_size)
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


    def forward(self, x):# 数据从此进来，经过定义好的各层网络，最终输出
        x = self.conv_unit(x)
        return x
    def forward_conv(self,x):
        x = self.conv_unit(x)
        return x


class triplet_loss(nn.Module): 
    def __init__(self,distance_function,margin): 
        super(triplet_loss, self).__init__() 
        self.margin = margin
        self.distance_function  = distance_function
        
    def forward(self, anchor, positive, negative): 
        torch.nn.CosineSimilarity
        pos_dist = self.distance_function(anchor,positive)
        neg_dist = self.distance_function(anchor,negative)
        output = torch.clamp(self.margin-pos_dist+neg_dist,min = 0)
        return output.mean()