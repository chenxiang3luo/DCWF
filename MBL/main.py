import torch.optim as optim
from model import DF
import numpy as np
import time
import configparser
import torch
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import time
import numpy as np
import torch as t
from model import DF
import shutil
from torchsummary import summary
import random
from util.data_generator import *
import torch.nn as nn
from dotmap import DotMap

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


class myDataset2(Dataset):
    def __init__(self,x,y):
        self.x_data = x
        self.y_data = y
        self.len = x.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

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

def update_config(config, updates):
    """Updates config dict and config file with updates dict."""
    config.update(updates)
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',savedir='.'):
    t.save(state, savedir+'/'+filename)
    if is_best:
        shutil.copyfile(savedir+'/'+filename, savedir+'/'+'model_best.pth.tar')

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


def train_meta(model):
    """Train and validate model."""
    args = DotMap()
    args.gpu = 0 
    args.num_shots_training = 5
    args.num_ways_training = 100
    args.max_train_iter = 8000
    args.num_workers = 16
    args.shot = 5
    args.way = 20
    args.query = 15
    args.batch_size_training = 15
    args.batch_size_inference = 128
    args.num_query_training = args.batch_size_training
    args.local_epochs = 100
    train_time_start = time.time()
    
    criterion = nn.CrossEntropyLoss()
    gpu = 0

    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
        criterion = criterion.cuda(gpu)
    for param in model.named_parameters():
        if not ('conv' in param[0] and 'bias' in param[0]):
            param[1].requires_grad = False

    optimizer_bias = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.0001)
    model.classifier = nn.Sequential(nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 100)).cuda()
    optimizer_classfier = optim.SGD(model.classifier.parameters(), lr=0.01)
    lr_scheduler = t.optim.lr_scheduler.StepLR(optimizer_bias,100,0.5)
    # for param in model.named_parameters():
    #     print(param[0],param[1].requires_grad)
    train_loader = get_base_dataloader_meta(args,"AWF_775")
    train_iterator = iter(train_loader) 
    for epoch in tqdm(range(args.max_train_iter), total=args.max_train_iter,desc='Epoch'):
        losses = AverageMeter('Loss')
        acc = AverageMeter('Acc@1')
        batch = next(train_iterator)
        data, train_label = batch[0],batch[1]

        train_label = [str(dd) for dd in train_label]
        support,query,support_label,query_label = train_test_split(data,train_label,test_size=args.query*args.way,stratify=train_label)
        le = MyLabelEncoder().fit(support_label)
        support_label = le.transform(support_label)
        query_label = le.transform(query_label)
        support_set = myDataset2(support,support_label)
        support_loader = DataLoader(support_set,batch_size=4)
        query_set = myDataset2(query,query_label)
        query_loader = DataLoader(query_set,batch_size=query_set.__len__())
        model.eval()
        # classifier
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight,0)
        
        for i in range(args.local_epochs):
            for j, batch in enumerate(support_loader):
                data, train_label = [_.cuda(gpu,non_blocking=True) for _ in batch]
                optimizer_classfier.zero_grad()
                output = model(data)
                loss = criterion(output,train_label)
                loss.backward()
                optimizer_classfier.step()
        model.zero_grad()

        # optimizer_classfier.zero_grad()
        # output = model(support)
        # loss = criterion(output,support_label)
        # loss.backward()
        # optimizer_classfier.step()
        # optimizer_classfier.zero_grad()
        
        # bias
        query_batch = next(iter(query_loader))
        query, query_label = [_.cuda(args.gpu,non_blocking=True) for _ in query_batch]
        optimizer_bias.zero_grad()
        output = model(query)
        loss = criterion(output,query_label)
        loss.backward()
        optimizer_bias.step()
        # optimizer_bias.zero_grad()

        lr_scheduler.step()
        # for param in model.named_parameters():
        #     if  'conv' in param[0]:
        #         print(param[1].grad)
        accuracy = top1accuracy(output.argmax(dim=1),query_label)
        losses.update(loss.item(),query.size(0))
        acc.update(accuracy.item(), query.size(0))
            
        print('training_loss/pretrain_CEL',losses.avg,epoch)
        print('accuracy/pretrain_train',acc.avg, epoch)
    torch.save(model.state_dict(), './pre_trained_model/finish_model_fine.pt')
    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))

def test_meta(model,shot):
    args = DotMap()
    args.gpu = 0 
    epochs = 100
    args.num_workers = 16
    args.query = 70
    args.way = 20
    args.shot = shot
    train_time_start = time.time()

    clf = nn.Sequential(nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 100)).cuda()
    gpu = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(),lr=0.01)
    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
    support_loader,query_loader = get_new_dataloader(args,'tor_100w_2500tr',100)
    model.eval()
    for epoch in tqdm(range(epochs),desc='Epoch'):
        clf.train(True)
        losses = AverageMeter('Loss')
        for i, batch in enumerate(support_loader):
            data, train_label = [_.cuda(gpu,non_blocking=True) for _ in batch]

        # forward pass
            optimizer.zero_grad()
            with t.no_grad():
                output = model.feature_forward(data)

            # forward pass
            output = clf(output)
            loss = criterion(output,train_label)

            # Backpropagation
            loss.backward()
            optimizer.step()
        
            losses.update(loss.item(),data.size(0))
        # if (epoch % 10 == 0):
        #     print('training_loss/pretrain_CEL',losses.avg,epoch)
        #     loo, acc ,tpr,fpr,f1 = validation_meta(model,clf,criterion,query_loader)  
        #     print('test: {}, loss: {}, TPR: {}, FPR: {}, F1: {}'.format(acc,loo,tpr,fpr,f1))
    print('training_loss/pretrain_CEL',losses.avg,epoch)
    loo, acc ,tpr,fpr,f1 = validation_meta(model,clf,criterion,query_loader)  
    print('test: {}, loss: {}, TPR: {}, FPR: {}, F1: {}'.format(acc,loo,tpr,fpr,f1))


        # lr_scheduler.step(val_loss)
    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))
def train_base(model):

    train_time_start = time.time()
    
    criterion = nn.CrossEntropyLoss()
    gpu = 0
    epochs = 110

    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
        criterion = criterion.cuda(gpu)

    optimizer = optim.SGD(model.parameters(),lr=0.1)
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.2)
    earlystop = EarlyStopping(5*2,0)
    train_loader = get_base_dataloader('AWF_775')
    for epoch in tqdm(range(epochs),desc='Epoch'):
        global_count = 0

        losses = AverageMeter('Loss')
        acc = AverageMeter('Acc@1')
        
        model.train(True)
        for i, batch in enumerate(train_loader):
            global_count = global_count + 1
            data, train_label = [_.cuda(gpu,non_blocking=True) for _ in batch]

            # forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,train_label)

            # Backpropagation
            loss.backward()
            optimizer.step()

            accuracy = top1accuracy(output.argmax(dim=1),train_label)

            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(), data.size(0))

        lr_scheduler.step()  

        print('training_loss/pretrain_CEL',losses.avg,epoch)
        print('accuracy/pretrain_train',acc.avg, epoch)
        torch.save(model.state_dict(), './pre_trained_model/finish_model_open.pt')

    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))


def validation(model,criterion,dataloader):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    pre = []
    target = []
    model.eval()
    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(0,non_blocking=True) for _ in batch]

            output = model(data)
            pre.extend(output.argmax(dim=1).cpu())
            target.extend(label.cpu())
            loss = criterion(output,label)
            accuracy = top1accuracy(output.argmax(dim=1),label)
            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(),data.size(0))
    pre = torch.tensor(pre,dtype=torch.float32)
    
    target = torch.tensor(target,dtype=torch.float32)
    acc_pre = pre.eq(target).float()
    acm = {}
    for label in np.unique(target):
        inds = np.argwhere(target == label)
        acm[label] = acc_pre[inds].sum().sum()/95
    # Plot figure if needed
    # print(acm)
    return losses.avg, acc.avg

def validation_meta(model,cls,criterion,dataloader):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    pre = []
    target = []
    model.eval()
    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(0,non_blocking=True) for _ in batch]

            output = model.feature_forward(data)
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
model = DF(775)
model.cuda()
summary(model,(1,5000))
# train_base(model)
model.load_state_dict(torch.load('./pre_trained_model/finish_model_open.pt'))
# train_meta(model)
# dic = torch.load('./pre_trained_model/finish_model_open.pt')
# to_zero = [key for key in dic.keys() if 'conv' in key and 'bias' in key]
# for zo in to_zero:
#     dic[zo] = torch.zeros_like(dic[zo],requires_grad=True)
# model.load_state_dict(dic)
for dataset,num_class in zip(['tor_100w_2500tr'],\
                             [100]):
    print('dataset',dataset)
    for shot in [1,5,10,15,20]:
        model.load_state_dict(torch.load('./pre_trained_model/finish_model_open.pt'))
        test_meta(model,shot)

