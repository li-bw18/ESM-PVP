import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, precision_recall_curve, average_precision_score
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
            self.x = np.array(pd.read_table("data/binary/train.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/binary/train_label.txt", header=None, index_col=None)).flatten()
        elif mode == 'valid':
            self.x = np.array(pd.read_table("data/binary/valid.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/binary/valid_label.txt", header=None, index_col=None)).flatten()
        else:
            self.x = np.array(pd.read_table("data/binary/test.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/binary/test_label.txt", header=None, index_col=None)).flatten()
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.long)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y
    def __len__(self):
        return len(self.x)




batch_size = 2
test_dataset = PVP(mode='test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
esmpvp = model.ESMPVP_binary(2)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    esmpvp = nn.DataParallel(esmpvp)
esmpvp = esmpvp.to(device)
esmpvp.load_state_dict(torch.load("model/binary.pth"))
all_label = np.array([])
all_predict = np.array([])
all_max = np.array([])

with torch.no_grad():
    esmpvp = esmpvp.eval()
    for batch_id, batch_data in enumerate(test_dataloader):
        inputs,labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicts = F.softmax(esmpvp(inputs)[4], 1).cpu().numpy()
        all_label = np.concatenate((all_label, labels.cpu().numpy()))
        all_predict = np.concatenate((all_predict, predicts[:, 1]))
        all_max = np.concatenate((all_max, np.argmax(predicts, axis=1)))
    
    test_acc = (all_label == all_max).sum() / len(all_label)
    test_auc = roc_auc_score(all_label, all_predict)
    print(f"ACC:{test_acc}")
    print(f"AUC:{test_auc}")
    p_class, r_class, f_class, _ = precision_recall_fscore_support(y_true=all_label, y_pred=all_max, average='binary')
    print(f"Precision:{p_class}")
    print(f"Recall:{r_class}")
    print(f"F1:{f_class}")
    test_ap = average_precision_score(all_label, all_predict)
    print(f"AP:{test_ap}")

