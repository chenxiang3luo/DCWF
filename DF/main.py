from model import DF
import numpy as np
import time
import random
import os
import argparse
import configparser

from tqdm import tqdm
import json
import os
import time
import numpy as np
import torch as t
from model import DF
import shutil
from data_generator import *
import torch.nn as nn


def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)  
    return dict(cf['default'])



def loadData(fpath):
    train = np.load(fpath,allow_pickle=True).item()
    train_X ,train_y = train['feature'], train['label']
    return train_X, train_y

def score_func(ground_truths, predictions):
    global MON_SITE_NUM
    tp, wp, fp, p, n = 0, 0, 0, 0 ,0
    for truth,prediction in zip(ground_truths, predictions):
        if truth != MON_SITE_NUM:
            p += 1
        else:
            n += 1
        if prediction != MON_SITE_NUM:
            if truth == prediction:
                tp += 1
            else:
                if truth != MON_SITE_NUM:
                    wp += 1
                    # logger.info('Wrong positive:%d %d'%(truth, prediction))
                else:
                    fp += 1
                    # logger.info('False positive:%d %d'%(truth, prediction))
    # logger.info('{} {} {} {} {}'.format(tp, wp, fp, p, n))
    try:
        r_precision = tp*n / (tp*n+wp*n+r*p*fp)
    except:
        r_precision = 0.0
    # logger.info('r-precision:%.4f',r_precision)
    # return r_precision
    return tp, wp, fp, p, n

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


def train_and_val( model,session,name):
    """Train and validate model."""

    # train_size = int(
    #     (num_mon_sites * num_mon_inst_train + num_unmon_sites_train) * 0.95)
    # val_size = int(
    #     (num_mon_sites * num_mon_inst_train + num_unmon_sites_train) * 0.05)

    # output_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1

    train_time_start = time.time()
    
    criterion = nn.CrossEntropyLoss()
    gpu = 0
    epochs = 150

    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
        criterion = criterion.cuda(gpu)

    optimizer = t.optim.Adamax(model.parameters(),lr=0.002)
    
    # lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=np.sqrt(0.1), patience=5,min_lr=1e-5)
    earlystop = EarlyStopping(5*2,0)
    print("Session: ",session)
    trainset, train_loader, val_loader = get_new_dataloader(session,name)
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

            
        print('training_loss/pretrain_CEL',losses.avg,epoch)
        print('accuracy/pretrain_train',acc.avg, epoch)
        val_loss, val_acc_mean = validation(model,criterion,val_loader)
        print('validation_loss/pretrain_CEL', val_loss,epoch)
        print('accuracy/pretrain_val', val_acc_mean,epoch)
        earlystop(val_loss)
        if earlystop.early_stop:
            break
        
        # lr_scheduler.step(val_loss)
    train_time_end = time.time()
    # test
    # test_loss,tset_acc_mean = validation(model,criterion,val_loader)
    # print('validation_loss/pretrain_CEL', test_loss)
    # print('accuracy/pretrain_val', tset_acc_mean)
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

# with open('config.json') as config_file:
#     config = json.load(config_file)



# if not os.path.exists('%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites,
#                                             num_mon_inst,
#                                             num_unmon_sites_train,
#                                             num_unmon_sites_test)):
#     preprocess_data.main(config)
tt = ['3d','10d','2w','4w']
for j in tt:
    print(j)
    for i in range(6):
        num = 100+i*20
        model = DF(num_classes=num)
        train_and_val(model,i,j)

# print('evaluating mixture on test data...')
# evaluate.main(config)


# if __name__ == "__main__":
#     cf = read_conf(const.confdir)
#     MON_SITE_NUM = int(cf['monitored_site_num'])
#     random.seed(0)
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#     EXP_Type = 'OpenWorld_NoDef'
#     # print ("Experimental Type: ", EXP_Type)
#     # network and training
#     NB_EPOCH = 20
#     # print ("Number of Epoch: ", NB_EPOCH)
#     BATCH_SIZE = 128
#     VERBOSE = 1
#     LENGTH = 106
#     OPTIMIZER = tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

#     NB_CLASSES = 101 # number of outputs: 100 Monitored websites + 1 Unmonitored websites
#     INPUT_SHAPE = (LENGTH,1)

 
#     sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
#     tps, wps, fps, ps, ns = 0, 0, 0, 0, 0
#     start_time = time.time()
#     folder_num = 1
#     for train_index, test_index in sss.split(X,y):
#         # logger.info('Testing fold %d'%folder_num)
#         folder_num += 1 
# #       if folder_num > 2:
# #           break
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
        
        
#         # initialize the optimizer and model
#         # print (time.sleep(2))
#         model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

#         model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
#             metrics=["accuracy"])
#         # print ("Model compiled")

#         # Start training
#         history = model.fit(X_train, y_train,
#                 batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,validation_split=0.1)

#         y_pred = model.predict(X_test)
#         y_pred = np.argmax(y_pred,axis = 1)
#         y_test = np.argmax(y_test, axis= 1)

#         tp, wp, fp, p, n = score_func(y_test, y_pred)
#         tps += tp
#         wps += wp
#         fps += fp     
#         ps += p
#         ns += n  
#         print("{:3d} {:3d} {:3d} {:3d} {:3d}".format(tp, wp, fp, p, n))
#     print("{:3d} {:3d} {:3d} {:3d} {:3d}".format(tps, wps, fps, ps, ns))
    # print("time:", time.time()-start_time)

