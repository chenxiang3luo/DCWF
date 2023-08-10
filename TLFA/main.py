import torch.optim as optim
from model import DF
import numpy as np
import time
import configparser
import torch
from tqdm import tqdm
import json
import time
from torchsummary import summary
import numpy as np
import torch as t
from model import DF
import shutil
import random
from util.data_generator import *
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from dotmap import DotMap
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)




def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)  
    return dict(cf['default'])



def loadData(fpath):
    train = np.load(fpath,allow_pickle=True).item()
    train_X ,train_y = train['feature'], train['label']
    return train_X, train_y


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


def train_novel( model,data_set,num_class,shot):
    """Train and validate model."""
    args = DotMap()
    args.gpu = 0 

    args.num_workers = 16
    args.query = 70
    args.shot = shot
    train_time_start = time.time()
    

    gpu = 0

    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
    model.eval()
    with t.no_grad():
        support_loader,query_loader = get_new_dataloader(args,data_set,num_class)

        support_data,support_labels = next(iter(support_loader))
        query_data,query_labels = next(iter(query_loader))

        support_feature = model.feature_forward(support_data.cuda())
        query_feature = model.feature_forward(query_data.cuda())
        
        # MLP
        clf = LinearSVC(random_state=42,max_iter=3000).fit(support_feature.cpu(), support_labels.cpu())
        predict = clf.predict(query_feature.cpu())
        tpr,fpr,f1 = get_matrix(predict,query_labels.cpu())
        print(clf.score(query_feature.cpu(),query_labels.cpu()),tpr.mean(),fpr.mean(),f1.mean())


        # lr_scheduler.step(val_loss)
    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))

def train_base(model):
    """Train and validate model."""


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
        # val_loss, val_acc_mean = validation(model,criterion,val_loader)
        # print('validation_loss/pretrain_CEL', val_loss,epoch)
        # print('accuracy/pretrain_val', val_acc_mean,epoch)
        torch.save(model.state_dict(), './pre_trained_model/finish_model.pt')

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


setup_seed(42)
model = DF(775)
model.cuda()
summary(model,(1,5000))
# train_base(model)
model.load_state_dict(torch.load('./pre_trained_model/finish_model.pt'))
for dataset,num_class in zip(['DF95','tor_100w_2500tr','KNN','DF19'],[95,100,100,100]):
    for shot in [1,5,10,15,20]:
        print('dataset:',dataset)
        print('shot:',shot)
        train_novel(model,dataset,num_class,shot)
# tt = ['3d','10d','2w','4w']
# for j in tt:
#     print(j)
#     for i in range(6):
#         num = 100+i*20
#         model = DF(num_classes=num)
#         train_and_val(model,i,j)

