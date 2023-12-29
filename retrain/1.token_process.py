import esm
import pandas as pd

# Change sequences into tokens

lis = ['train', 'valid', 'test']
num = ['40', '50', '60', '70', '80', '90']

alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]
batch_converter = alphabet.get_batch_converter()

for li in lis:
    data = []
    with open(f"data/binary/{li}_raw.txt") as f:
        while 1:
            name = f.readline()
            if name == '':
                break
            data.append((name[:-1], f.readline()[:-1]))
    batch_tokens = pd.DataFrame(batch_converter(data)[2]).iloc[:, :1024]
    batch_tokens.to_csv(f"data/binary/{li}.txt", sep='\t', header=False, index=False)
    print(f"Finish: binary, {li}")

for li in lis:
    for n in num:
        data = []
        with open(f"data/multi/d{n}/{li}_raw.txt") as f:
            while 1:
                name = f.readline()
                if name == '':
                    break
                data.append((name[:-1], f.readline()[:-1]))
        batch_tokens = pd.DataFrame(batch_converter(data)[2]).iloc[:, :1024]
        batch_tokens.to_csv(f"data/multi/d{n}/{li}.txt", sep='\t', header=False, index=False)
        print(f"Finish: multi, {n}, {li}")
