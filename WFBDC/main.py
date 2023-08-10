import torch.optim as optim
from util.model import DF,get_matrix
import numpy as np
import time
import torch
from tqdm import tqdm
import time
from torchsummary import summary
import numpy as np
import torch as t
import random
from util.MS_Loss import *
from util.data_generator import *
import torch.nn as nn
from sklearn.svm import LinearSVC
from dotmap import DotMap
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
            not improving
        :param min_delta: minimum difference between new loss and old loss for
            new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def top1accuracy(pred, target):
    """Computes the precision@1"""
    batch_size = target.size(0)

    correct = pred.eq(target).float().sum(0)
    return correct.mul_(100.0 / batch_size)

def is_valid_mixture(mixture):
    """Check if mixture is a 2D array with strings representing the models."""
    assert type(mixture) == list and len(mixture) > 0
    for inner_comb in mixture:
        assert type(inner_comb) == list and len(inner_comb) > 0
        for model in inner_comb:
            assert model in ['dir', 'time', 'metadata']


def train_novel( model,num_classes,shot = 1,dataset = None,include=0):
    """Train and validate model."""
    args = DotMap()
    args.gpu = 0 
    epochs = 500
    args.num_workers = 16
    args.query = 70
    args.shot = shot
    train_time_start = time.time()
    
    clf = nn.Sequential(nn.Linear(32896,num_classes)).cuda()
    gpu = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(),lr=0.01)
    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
    # support_loader,query_loader = get_new_dataloader(args,dataset,num_classes)
    support_loader,query_loader = get_new_dataloader(args,dataset,num_classes)
    model.eval()
    for epoch in tqdm(range(epochs),desc='Epoch'):
        clf.train(True)
        losses = AverageMeter('Loss')
        for i, batch in enumerate(support_loader):
            data, train_label = [_.cuda(gpu,non_blocking=True) for _ in batch]

        # forward pass
            optimizer.zero_grad()
            with t.no_grad():
                output = model(data)

            # forward pass
            output = clf(output)
            if(train_label.shape[0] == 1):
                continue
            loss = criterion(output,train_label)

            # Backpropagation
            loss.backward()
            optimizer.step()
        
            losses.update(loss.item(),data.size(0))
        if (epoch % 50 == 0):
            print('training_loss/pretrain_CEL',losses.avg,epoch)
            loo, acc ,tpr,fpr,f1 = validation(model,clf,criterion,query_loader)   
            print('test: {}, loss: {}, TPR: {}, FPR: {}, F1: {}'.format(acc,loo,tpr,fpr,f1))
    
    torch.save(clf.state_dict(), f'./pre_trained_model/open_model_{dataset}_{shot}.pt')
    print('training_loss/pretrain_CEL',losses.avg,epoch)
    loo, acc ,tpr,fpr,f1 = validation(model,clf,criterion,query_loader)   
    print('test: {}, loss: {}, TPR: {}, FPR: {}, F1: {}'.format(acc,loo,tpr,fpr,f1))


        # lr_scheduler.step(val_loss)
    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))

def train_base(model):
    """Train and validate model."""


    train_time_start = time.time()
    
    criterion = MultiSimilarityLoss()
    gpu = 0
    epochs = 8000

    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
        criterion = criterion.cuda(gpu)

    optimizer = optim.SGD(model.parameters(),lr=0.1)
    train_loader = get_base_dataloader()
    train_iterator = iter(train_loader)
    for epoch in tqdm(range(epochs),desc='Epoch'):
        if(epoch % 2000 == 0):
            s = (epoch//2000) * 10
            criterion.updata_scale(8+s,128 - s)
        losses = AverageMeter('Loss')
        
        model.train(True)
        batch = next(train_iterator)
        data, train_label = [_.cuda(gpu,non_blocking=True) for _ in batch]
        # forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,train_label)

        # Backpropagation
        loss.backward()
        optimizer.step()

        losses.update(loss.item(),data.size(0))
        
        
        if (epoch % 1000 == 0):
            print('training_loss/pretrain_CEL',losses.avg,epoch)
            torch.save(model.state_dict(), './pre_trained_model/finish_model_BDC.pt')
    print('training_loss/pretrain_CEL',losses.avg,epoch)
    torch.save(model.state_dict(), './pre_trained_model/finish_model_BDC.pt')
    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))

def validation(model,cls,criterion,dataloader):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    pre = []
    target = []
    model.eval()
    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(0,non_blocking=True) for _ in batch]

            output = model(data)
            output = cls(output)
            pre.extend(output.argmax(dim=1).cpu())
            target.extend(label.cpu())
            loss = criterion(output,label)
            accuracy = top1accuracy(output.argmax(dim=1),label)
            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(),data.size(0))
    pre = torch.tensor(pre,dtype=torch.float32)
    
    target = torch.tensor(target,dtype=torch.float32)
    TPR,FPR,F1 = get_matrix(pre,target)
    # acc_pre = pre.eq(target).float()
    # acm = {}
    # for label in np.unique(target):
    #     inds = np.argwhere(target == label)
    #     acm[label] = acc_pre[inds].sum().sum()/95
    # Plot figure if needed
    # print(acm)
    return losses.avg, acc.avg,TPR.mean(),FPR.mean(),F1.mean()


setup_seed(42)
model = DF()
model.cuda()
summary(model,(1,5000))
# train_base(model)
model.load_state_dict(torch.load('./pre_trained_model/finish_model_BDC.pt'))
# for dataset,num_class in zip(['DF19','DF95'],[100,95]):
#     print(dataset)
#     for shot in [1,5,10,15,20]:
#         print('shot:',shot)
#         train_novel(model,num_class,shot,dataset)
# for dataset in ['tor_time_test3d_200w_100tr','tor_time_test10d_200w_100tr','tor_time_test2w_200w_100tr','tor_time_test4w_200w_100tr','tor_time_test6w_200w_100tr']:
#     print(dataset)
#     for shot in [1,5,10,15,20]:
#         print('shot:',shot)
#         train_novel(model,num_class,shot,dataset)

# num_class = 100
for dataset,num_class in zip(['DF95','tor_100w_2500tr','KNN','DF19'],[95,100,100,100]):
    for shot in [1,5,10,15,20]:
        print('dataset:',dataset)
        print('shot:',shot)
        train_novel(model,num_class,shot,dataset)
# print('open Scenario')
# for dataset,num_class in zip(['tor_100w_2500tr'],[100]):
#     print(dataset)
#     for shot in [1,5,10,15,20]:
#         print('shot:',shot)
#         train_novel(model,num_class,shot,dataset)