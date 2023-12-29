import torch.nn as nn
import torch.nn.functional as F
import torch
import esm

class ESMPVP_binary(nn.Module):
    def __init__(self, n_class, lis=['29', '30', '31', '32']):
        super(ESMPVP_binary, self).__init__()
        self.esm = esm.pretrained.esm2_t33_650M_UR50D()[0]
        for na, p in self.named_parameters():
            sp = na.split('.')
            if len(sp) > 2 and sp[2] not in lis:
                p.requires_grad = False
            else:
                p.requires_grad = True
        self.fc1 = nn.Linear(1280, 320)
        self.fc2 = nn.Linear(320, 80)
        self.fc3 = nn.Linear(80, 20)
        self.fc4 = nn.Linear(20, n_class)
    def forward(self, batch_tokens):
        x = self.esm(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        batch_tokens = batch_tokens.unsqueeze(-1)
        x = F.relu(x)
        x = x.masked_fill(batch_tokens==1, 0)[:, 1:, :]
        x = torch.sum(x, dim=1)
        ret = []
        ret.append(x)
        x = F.relu(self.fc1(x))
        ret.append(x)
        x = F.relu(self.fc2(x))
        ret.append(x)
        x = F.relu(self.fc3(x))
        ret.append(x)
        ret.append(self.fc4(x))
        return ret


class ESMPVP_multi(nn.Module):
    def __init__(self, n_class):
        super(ESMPVP_multi, self).__init__()
        self.fc1 = nn.Linear(1280, 320)
        self.bn1 = nn.BatchNorm1d(320)
        self.fc2 = nn.Linear(320, 80)
        self.bn2 = nn.BatchNorm1d(80)
        self.fc3 = nn.Linear(80, 20)
        self.bn3 = nn.BatchNorm1d(20)
        self.fc4 = nn.Linear(20, n_class)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)
