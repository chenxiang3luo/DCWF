from DF_model_Torch import DF
import torch
from torch.utils.data import Dataset, DataLoader
# from data_generator import MyLabelEncoder,myDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
import pandas as pd
class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self
    

num = 900
model = DF(512)
model.cuda()
model.load_state_dict(torch.load('./trained_model/embedding_model.pt'))
model.eval()
domain_list = ['DF19','AWF_novel','AWF_base','tor_time_test3d_200w_100tr']
with torch.no_grad():
    for domain in domain_list:
        dataset = np.load('/root/datasets/FSCIL/'+domain+'.npz',allow_pickle=True)
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
        np.save(f'/root/datasets/FSCIL/features/{domain}_feature_TP.npy', np.array(features))
        np.save(f'/root/datasets/FSCIL/features/{domain}_labels_TP.npy', np.array(labeles))