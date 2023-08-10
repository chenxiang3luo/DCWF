import torch
from pytorch_metric_learning import miners, losses

class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 8
        self.scale_neg = 128
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
    
    def forward(self, embeddings, labels):
        #难样本挖掘
        hard_pairs = self.miner(embeddings, labels) 
        #损失计算
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
    def updata_scale(self,scale_pos,scale_neg):
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg
        self.loss_func = losses.MultiSimilarityLoss(scale_pos, scale_neg, self.thresh)