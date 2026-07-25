import torch
from ASGAE import AE
import numpy as np
from pathlib import Path
from os import path

torch.backends.cudnn.deterministic = True
seed = 472
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

emb_dims = 512

# dataloadi
num_workers = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
datadir = '/mnt/d/OneDrive - The University of Colorado Denver/DataMeshCHCO4_25'
# datadir = r'D:\OneDrive - The University of Colorado Denver\DataMeshCHCO4_25'
files = [str(x) for x in Path(datadir).glob('*_meshHeadTexLand.pt')]

trainDict={
    'batch_size': 4,
    'test_batch_size': 2,
    'shuffle': True,
    'setSplit': [0.8, 0.1, 0.1]
}
net = AE(6,emb_dims, bias=True, Dataset=files, trainDic=trainDict)
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
for i in range(10):
    net.Train_one_epoch(optimizer, device)
    net.Test_one_epoch(device)