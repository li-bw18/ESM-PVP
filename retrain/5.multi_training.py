import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
import model


# Determine which GPU will be used
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
            self.x = np.array(pd.read_table("data/multi/prompt/d40/train.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/multi/prompt/d40/train_label.txt", header=None, index_col=None)).flatten()
        elif mode == 'valid':
            self.x = np.array(pd.read_table("data/multi/prompt/d40/valid.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/multi/prompt/d40/valid_label.txt", header=None, index_col=None)).flatten()
        else:
            self.x = np.array(pd.read_table("data/multi/prompt/d40/test.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/multi/prompt/d40/test_label.txt", header=None, index_col=None)).flatten()
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y
    def __len__(self):
        return len(self.x)

def trail(model, train_dataloader, val_dataloader, criterion, optimizer, nepoch, path):
    current_highest_acc = -1
    current_best_epoch = 0
    train_loss_list = []
    valid_loss_list = []
    valid_acc_list = []
    for epoch_id in range(nepoch):
        model.train()
        train_loss = 0
        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs,labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            predicts = F.softmax(model(inputs), 1)
            loss = criterion(predicts, labels)
            train_loss += loss.cpu().detach().numpy()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss)
        valid_loss = 0 
        valid_acc = 0
        valid_num = 0
        with torch.no_grad():
            model.eval()
            for batch_id, batch_data in enumerate(val_dataloader):
                inputs,labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                predicts = F.softmax(model(inputs), 1)
                valid_loss += criterion(predicts, labels).cpu().detach().numpy()
                predicts = predicts.cpu().numpy()
                valid_num += len(labels)
                valid_acc += (labels.cpu().numpy() == np.argmax(predicts, axis=1)).sum()
        valid_loss = valid_loss / len(val_dataloader)
        valid_loss_list.append(valid_loss)
        valid_acc = valid_acc / valid_num
        if valid_acc > current_highest_acc:
            current_highest_acc = valid_acc
            current_best_epoch = epoch_id
            torch.save(model.state_dict(), path)
        print("Epoch:", epoch_id)
        print("--------------------") 
        print("Train loss:", train_loss)
        print("Valid loss:", valid_loss)
        print("Valid ACC:", valid_acc)
        print("--------------------") 
        if epoch_id - current_best_epoch == 10:
            break
    print("Best Valid ACC:", current_highest_acc)  
    print("Best Epoch:", current_best_epoch)          
    print("--------------------") 
    return train_loss_list, valid_loss_list, valid_acc_list 

batch_size = 64
learning_rate = 1e-4
weight_decay = 1e-5
nepoch = 100

weight = torch.tensor([1.239, 4.716, 3.045, 12.87, 1, 1.654, 1.933],dtype=torch.float) # 0.4 weight
# weight = torch.tensor([1.322, 2.086, 3.275, 13.06, 1, 1.792, 2.242],dtype=torch.float) # 0.5 weight
# weight = torch.tensor([1.421, 2.075, 3.428, 13.29, 1, 1.707, 2.187],dtype=torch.float) # 0.6 weight
# weight = torch.tensor([1.329, 1.994, 3.498, 13.21, 1, 1.856, 2.123],dtype=torch.float) # 0.7 weight
# weight = torch.tensor([1.411, 2.031, 3.251, 13.13, 1, 1.813, 2.054],dtype=torch.float) # 0.8 weight
# weight = torch.tensor([1.383, 1.977, 3.421, 13.13, 1, 1.802, 2.087],dtype=torch.float) # 0.9 weight
esmpvp = model.ESMPVP_multi(7)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    esmpvp = nn.DataParallel(esmpvp)
esmpvp = esmpvp.to(device)
train_dataset = PVP(mode='train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = PVP(mode='valid')
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss(weight=weight).to(device)
optimizer = torch.optim.Adam(esmpvp.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loss_list, valid_loss_list, valid_acc_list = trail(esmpvp, train_dataloader, val_dataloader, criterion, optimizer, nepoch, "model/multi_model_40.pth")

