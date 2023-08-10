from math import pi
from math import cos
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model.model_torch import *
import numpy as np
from util.data_loader import *
# variables


cuda = torch.cuda.is_available()
batch_size = 64



def proposed_lr(initial_lr, iteration, epoch_per_cycle):
    # proposed learning late function
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2

def train_DEA(model,epochs):
    early_stop = EarlyStopping(patience=10)
    train_loader,test_loader = get_dataloader(0)
    optimizer = optim.NAdam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for j in range(epochs):
        losses = AverageMeter('Loss')
        acc = AverageMeter('Acc@1')
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(data,output)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(),data.size(0))
            early_stop(loss.item(),model)
    return 


def train_se(model,epochs,cycles,initial_lr):
    """
    during an iteration a batch goes forward and backward  
    while during an epoch every batch of a data set is processed
    """
    train_loader,test_loader = get_dataloader(0)
    snapshots = []
    _lr_list = []
    count = 0
    epochs_per_cycle = epochs // cycles
    optimizer = optim.NAdam(model.parameters(), lr=initial_lr,weight_decay=0.004)
    criterion = nn.CrossEntropyLoss()
    for i in range(cycles):

        for j in range(epochs_per_cycle):
            losses = AverageMeter('Loss')
            acc = AverageMeter('Acc@1')

            lr = proposed_lr(initial_lr, j, epochs_per_cycle)
            optimizer.state_dict()["param_groups"][0]["lr"] = lr

            for batch_idx, (data, target) in enumerate(train_loader):
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                accuracy = top1accuracy(output.argmax(dim=1),target)
                acc.update(accuracy.item(), data.size(0))
                losses.update(loss.item(),data.size(0))

            _lr_list.append(lr)
            count += 1
            validation(model,criterion,test_loader)

        snapshots.append(model.state_dict())
    return snapshots


def test_se(Model, snapshots, use_model_num):
    train_loader,test_loader = get_dataloader(0)
    index = len(snapshots) - use_model_num
    snapshots = snapshots[index:]
    model_list = [Model() for _ in snapshots]

    for model, weight in zip(model_list, snapshots):
        model.load_state_dict(weight)
        model.eval()
        if cuda:
            model.cuda()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output_list = [model(data).unsqueeze(0) for model in model_list]
        output = torch.mean(torch.cat(output_list), 0).squeeze()
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)))

    return test_loss


def top1accuracy(pred, target):
    """Computes the precision@1"""
    batch_size = target.size(0)
    
    correct = pred.eq(target).float().sum(0)
    return correct.mul_(100.0 / batch_size)
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


def validation(model,criterion,dataloader):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    pre = []
    target = []
    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(0,non_blocking=True) for _ in batch]

            output = model(data)
            pre.extend(output)
            target.extend(label)
            loss = criterion(output,label)
            accuracy = top1accuracy(output.argmax(dim=1),label)
            
            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(),data.size(0))
    # Plot figure if needed
    # pre = torch.tensor(pre,dtype=torch.float32)
    # target = torch.tensor(target,dtype='float32')
    # acc_pre = pre.eq(target).float()
    # acm = {}
    # for label in np.array(target).unique():
    #     inds = np.argwhere(target == label)
    #     acm[label] = acc_pre[inds].sum(0)
    # print(acm)
    return losses.avg, acc.avg
if __name__ == '__main__':
    model = DAE()
    if cuda:
        model.cuda()
    
    # if cuda:
    #     model.cuda()
    # print("snapshot ensemble")
    # models = train_se(model, 100, 6, 0.002)
    # test_se(DF, models, 5)