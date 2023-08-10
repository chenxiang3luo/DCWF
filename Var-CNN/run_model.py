from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import os
import time
import numpy as np
import torch as t
from var_cnn import ResNet18
import shutil
# import evaluate
# import preprocess_data
from data_generator import *
# from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical
import torch.nn as nn

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
    def __call__(self, val_loss,model):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            torch.save(model.state_dict(), './pre_trained_model/embedding_model.pt')
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


def train_and_val(model):
    """Train and validate model."""
    epoches = 110

    train_time_start = time.time()
    
    criterion = nn.CrossEntropyLoss()
    gpu = 0

    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
        criterion = criterion.cuda(gpu)

    optimizer = t.optim.Adam(model.parameters(),lr=1e-3)
    
    lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=np.sqrt(0.1), patience=5,min_lr=1e-5)
    earlystop = EarlyStopping(5*2,0)
    # for session in range(6):

    train_loader, val_loader = get_dataloader()
    for epoch in tqdm(range(epoches),desc='Epoch'):
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
        print('training_loss/pretrain_CEL',losses.avg,epoch)
        print('accuracy/pretrain_train',acc.avg, epoch)
        val_loss, val_acc_mean = validation(model,criterion,val_loader)
        print('validation_loss/pretrain_CEL', val_loss,epoch)
        print('accuracy/pretrain_val', val_acc_mean,epoch)
        earlystop(val_loss,model)
        if earlystop.early_stop:
            break
        
        lr_scheduler.step(val_loss)
    train_time_end = time.time()
    # test
    # test_loss,tset_acc_mean = validation(model,criterion,val_loader)
    # print('validation_loss/pretrain_CEL', test_loss)
    # print('accuracy/pretrain_val', tset_acc_mean)
    print('Total training time: %f' % (train_time_end - train_time_start))



def validation(model,criterion,dataloader):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(0,non_blocking=True) for _ in batch]

            output = model(data)
            loss = criterion(output,label)
            accuracy = top1accuracy(output.argmax(dim=1),label)
            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(),data.size(0))

    return losses.avg, acc.avg


num = 900
model = ResNet18(num)
train_and_val(model)

# print('evaluating mixture on test data...')
# evaluate.main(config)
