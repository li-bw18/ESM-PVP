import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import model


# Determine which GPU will be used
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print(torch.cuda.device_count())
print(torch.cuda.current_device())
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 2024
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class PVP(Dataset):
    def __init__(self, mode='train'):
        super(PVP, self).__init__()
        if mode == 'train':
            self.x = np.array(pd.read_table("data/multi/d40/train.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/multi/d40/train_label.txt", header=None, index_col=None)).flatten()
        elif mode == 'valid':
            self.x = np.array(pd.read_table("data/multi/d40/valid.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/multi/d40/valid_label.txt", header=None, index_col=None)).flatten()
        else:
            self.x = np.array(pd.read_table("data/multi/d40/test.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/multi/d40/test_label.txt", header=None, index_col=None)).flatten()
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.long)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y
    def __len__(self):
        return len(self.x)

batch_size=2
train_dataset = PVP(mode='train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = PVP(mode='valid')
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = PVP(mode='test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

esmpvp = model.ESMPVP_binary(2)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    esmpvp = nn.DataParallel(esmpvp)
esmpvp.load_state_dict(torch.load("model/binary.pth"))
esmpvp = esmpvp.to(device)



with torch.no_grad():
    esmpvp.eval()

    train = pd.DataFrame(data=None, columns=range(1280))
    train_labels = pd.DataFrame(data=None, columns=[0])
    for batch_id, batch_data in enumerate(train_dataloader):
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = pd.DataFrame(labels, columns=[0])
        predicts = pd.DataFrame(data=esmpvp(inputs)[0].cpu().detach().numpy(), columns=range(1280))
        train = pd.concat([train, predicts])
        train_labels = pd.concat([train_labels, labels])
    train.to_csv("data/multi/prompt/d40/train.txt", sep='\t', header=False, index=False)
    train_labels.to_csv("data/multi/prompt/d40/train_label.txt", sep='\t', header=False, index=False)

    valid = pd.DataFrame(data=None, columns=range(1280))
    valid_labels = pd.DataFrame(data=None, columns=[0])
    for batch_id, batch_data in enumerate(valid_dataloader):
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = pd.DataFrame(labels, columns=[0])
        predicts = pd.DataFrame(data=esmpvp(inputs)[0].cpu().detach().numpy(), columns=range(1280))
        valid = pd.concat([valid, predicts])
        valid_labels = pd.concat([valid_labels, labels])
    valid.to_csv("data/multi/prompt/d40/valid.txt", sep='\t', header=False, index=False)
    valid_labels.to_csv("data/multi/prompt/d40/valid_label.txt", sep='\t', header=False, index=False)

    test = pd.DataFrame(data=None, columns=range(1280))
    test_labels = pd.DataFrame(data=None, columns=[0])
    for batch_id, batch_data in enumerate(test_dataloader):
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = pd.DataFrame(labels, columns=[0])
        predicts = pd.DataFrame(data=esmpvp(inputs)[0].cpu().detach().numpy(), columns=range(1280))
        test = pd.concat([test, predicts])
        test_labels = pd.concat([test_labels, labels])
    test.to_csv("data/multi/prompt/d40/test.txt", sep='\t', header=False, index=False)
    test_labels.to_csv("data/multi/prompt/d40/test_label.txt", sep='\t', header=False, index=False)

