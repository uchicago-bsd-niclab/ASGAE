from torch_geometric.data.dataset import Dataset
import numpy as np
from pathlib import Path
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch
import app.util as utl
from torch_geometric.loader import DataLoader

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        #along y we want a new dimension to handle the spherical maps
        if key == 'y':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
    def __len__(self):
        return 1

class LoadedDataset(Dataset):
    def __init__(self, filenames,transform=False, translate=False):
        super().__init__()
        self.transform = transform
        self.translate = translate
        self.filenames =filenames

    def __getitem__(self, idx):
        if self.transform:
            data = self.load_file(self.filenames[idx])
            data = self.applyTransform(data, self.translate)
            return data
        else:
            return self.load_file(self.filenames[idx])

    def load_file(self, filename):
        # pdb.set_trace()
        if type(filename) == str:
            data = torch.load(filename, weights_only=False)
        return data
    
    def get_MaxDist(self):
        MaxDist = 0
        for i in range(len(self.filenames)):
            data1 = self.load_file(self.filenames[i])
            dist = torch.max(torch.norm(data1.pos,dim=1))
            if dist > MaxDist:
                MaxDist = dist            
        return MaxDist

    def get(self):
        pass
                
    def len(self):
        return len(self.filenames)
    
    def applyTransform(self, data, translate=True):
        # Apply a transformation to the data
        x = data.pos
        rad = 1/6
        R = utl.euler2rot(torch.Tensor(np.random.uniform(-rad,rad,3)*np.pi))
        trns = torch.Tensor(np.random.uniform(-50,50,3))
        if translate:
            pos2 = torch.matmul((x-torch.mean(x,axis=0)), R.transpose(1,0))+trns+torch.mean(x,axis=0)
        else:
            pos2 = torch.matmul((x-torch.mean(x,axis=0)), R.transpose(1,0))+torch.mean(x,axis=0)
        data.pos = pos2
        return data
        
    

class LoadedDatasetCOMA(Dataset):
    def __init__(self, filenames1,filenames2=None, normalize=False, dummy_node= False):
        super().__init__()
        
        self.filenames1 =filenames1
        self.filenames2 =filenames2
        self.normalize =normalize
        self.dummy_node =dummy_node
        if len(filenames1)==len(filenames2):
            self.pair = True
        else:
            self.pair = False
            # print("The dimensions in the input 1 and 2 are different, will process only with the first input")
        if self.normalize:
            self.meanR= torch.load('/mnt/c/Users/cruzguea/Documents/DataMeshCOMA/mean_R.pt', weights_only=False)
            self.stdR= torch.load('/mnt/c/Users/cruzguea/Documents/DataMeshCOMA/std_R.pt', weights_only=False)
            self.meanT= torch.load('/mnt/c/Users/cruzguea/Documents/DataMeshCOMA/mean_T.pt', weights_only=False)
            self.stdT= torch.load('/mnt/c/Users/cruzguea/Documents/DataMeshCOMA/std_T.pt', weights_only=False)

    def __getitem__(self, idx):
        if self.pair:
            return self.load_file(self.filenames1[idx]), self.load_file(self.filenames2[idx])
        else:
            return self.load_file(self.filenames1[idx])
        
    def load_file(self, filename):
        if type(filename) == str:
            data = torch.load(filename, weights_only=False)
            if self.normalize:
                if filename[-4]=='R':
                    data.x = (data.x-self.meanR)/self.stdR
                    data.y = data.x
                else:
                    data.x = (data.x-self.meanT)/self.stdT
                    data.y = data.x
            if self.dummy_node:
                data.x= torch.cat((data.x, torch.zeros(1, 3)), dim=0)
                data.num_nodes+=1
        return data

    def get(self):
        pass
    def len(self):
        return len(self.filenames1)
    
    
def GetDataLoaders(dataset, batch_size, batchtest=1, train_set_percentage=0.8, val_set_percentage=0.1, shuffle=False, num_workers=0, pin_memory=True):
    train_data, val_data, test_data = torch.utils.data.random_split(dataset,[train_set_percentage,val_set_percentage,1-(train_set_percentage+val_set_percentage)], torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, shuffle=False,  batch_size=batchtest, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, shuffle=False,batch_size=batchtest, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader