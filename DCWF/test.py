import torch.optim as optim
from model import DF
import numpy as np
import seaborn
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
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from dotmap import DotMap

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(42)
model = DF(100)
model.cuda()
model.hook_register()
model.load_state_dict(torch.load('./pre_trained_model/finish_model.pt'))
# summary(model,(1,5000))
train_dataset = np.load('/root/datasets/FSCIL/AWF_base.npz',allow_pickle=True)
x = train_dataset['data']
model(torch.tensor(x[2500],dtype=t.float32).cuda().reshape(1,1,5000))

print(model.activation)
model.reset_attention()