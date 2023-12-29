import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import retrain.model as model
from collections import OrderedDict
import utils
import subprocess

parser = argparse.ArgumentParser(description='ESM-PVP binary (identification of phage virion proteins)')
parser.add_argument('input', help='Path to the input fasta file')
parser.add_argument('-o', '--output', help='Path to output directory', default=f'{os.getcwd()}/result')
parser.add_argument('-g', '--gpu_id', help='Give the IDs of GPUs you want to use [For example:"-g 0,1"], if not provided, cpu will be used', default=None)
parser.add_argument('-b', '--batch_size', type=int, help='Define the batch size used in the prediction', default=2)
args = parser.parse_args()

seed = 2024
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
batch_size = args.batch_size
output = args.output

if os.path.exists(output) is False:
    subprocess.call([f"mkdir {output}"], shell=True)

names = utils.token_generator(args.input, output)
dataset = utils.PVP_binary(output)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

esmpvp = model.ESMPVP_binary(2)
if device != 'cpu' and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    esmpvp = nn.DataParallel(esmpvp)
    esmpvp.load_state_dict(torch.load("model/binary.pth"))
else:
    state_dict = torch.load("model/binary.pth", map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    esmpvp.load_state_dict(new_state_dict)
esmpvp = esmpvp.to(device)

print('Process 2: Prediction')
with torch.no_grad():
    esmpvp.eval()
    all_predict = []
    all_max = []
    for batch_id, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        predicts = F.softmax(esmpvp(batch_data)[4], 1).cpu().numpy()
        all_predict.append(predicts[:, 1])
        all_max.append(np.argmax(predicts, axis=1))
    all_predict = np.concatenate(all_predict)
    all_max = np.concatenate(all_max)
    with open(f'{output}/result_discription.txt', 'w') as f:
        f.write('Label information:\n')
        f.write('1: PVP\n')
        f.write('0: non-PVP\n')
        f.write('File information:\n')
        f.write('result.txt: column 1, sequence name; column 2, predicted label; column 3, probability\n')
        f.write('other files are useless\n')
    result = pd.DataFrame(data=None)
    result['pred'] = all_max
    result['prob'] = all_predict
    result.index = names
    result.to_csv(f'{output}/result.txt', sep='\t', header=True, index=True)    
print('Process 2 finished!')
print('All processes finished!')
