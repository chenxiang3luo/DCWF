from var_cnn import ResNet18
import torch
from torch.utils.data import Dataset, DataLoader
from data_generator import MyLabelEncoder,myDataset
import numpy as np
num = 900
model = ResNet18(num)
model.cuda()
model.load_state_dict(torch.load('./pre_trained_model/embedding_model.pt'))
model.eval()
domain_list = ['AWF_775_complete']
for domain in domain_list:
    dataset = np.load('/root/datasets/FSCIL/'+domain+'.npz',allow_pickle=True)
    data = dataset['data']
    labels = dataset['labels']
    labels = [str(lab) for lab in labels]
    tran = MyLabelEncoder()
    tran.fit(labels)
    labels = tran.transform(labels)
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
            output = model.feature_forward(datad)
            features.extend(output.cpu().numpy())
            labeles.extend(np.array(label))
    np.save(f'/root/datasets/FSCIL/features/{domain}_feature.npy', np.array(features))
    np.save(f'/root/datasets/FSCIL/features/{domain}_labels.npy', np.array(labeles))