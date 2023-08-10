import numpy as np
file_name = '/root/datasets/FSCIL/'+'AWF_775'+'.npz'
train_novel = np.load(file_name,allow_pickle=True)
train_data_775 = train_novel['data']
train_labels_775 = train_novel['labels']
classes = np.unique(train_labels_775)
file_name = '/root/datasets/FSCIL/'+'tor_900w_2500tr'+'.npz'
train_novel = np.load(file_name,allow_pickle=True)
train_data_900 = train_novel['data']
train_labels_900 = train_novel['labels']
cla = [c[19:] for c in classes]
data = []
label = []
for novel_class in cla:
    inds = np.argwhere(train_labels_900==novel_class)
    inds = inds.reshape(-1)
    data.extend(train_data_900[inds])
    label.extend(train_labels_900[inds])
data = np.array(data)
label = np.array(label)
np.savez('AWF_775_complete.npz',data=data,labels=label)