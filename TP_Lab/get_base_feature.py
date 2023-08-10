from DF_model_Torch import DF
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
import pandas as pd
class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self
    

model = DF(64)
model.cuda()
model.load_state_dict(torch.load('./trained_model/AWF_775_model.pt'))
model.eval()
base_domain = 'AWF_775'
with torch.no_grad():
    dataset = np.load('/root/datasets/FSCIL/'+base_domain+'.npz',allow_pickle=True)
    data = dataset['data']
    labels = dataset['labels']
    # tran = MyLabelEncoder()
    # tran.fit(labels)
    # labels = tran.transform(labels)
    data = data.reshape(data.shape[0],1,data.shape[1])
    dataset= list(zip(data,labels))
    trainloader = DataLoader(dataset=dataset, batch_size=512, shuffle=False,
                                            num_workers=16)
    features = []
    labeles = []
    with torch.no_grad(): 
        for i, batch in enumerate(trainloader):
            datad,label = batch
            datad = torch.tensor(datad,dtype=torch.float32).cuda()
            output = model(datad)
            features.extend(output.cpu().numpy())
            labeles.extend(np.array(label))

features = np.array(features)
labeles = np.array(labeles)
feature_dict = defaultdict(int)
print(np.unique(labeles).shape)
for label in np.unique(labeles):
    inds = np.argwhere(labeles==label)
    samples = np.random.permutation(len(inds))
    train_samples_base = inds[samples].reshape(-1)
    feature_dict[label] = features[train_samples_base]

with open(f'./{base_domain}_feature.pkl','wb') as f:
    pickle.dump(feature_dict,f)
