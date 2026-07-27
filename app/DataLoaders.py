"""Dataset and dataloader utilities for ASGAE.

Defines the PyTorch Geometric ``Data`` subclass used to store surface meshes,
the datasets that load them from ``.pt`` files (optionally applying random
rigid augmentation), and a helper to build train/validation/test dataloaders.
"""
from torch_geometric.data.dataset import Dataset
import numpy as np
from pathlib import Path
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch
import app.util as utl
from torch_geometric.loader import DataLoader

class MyData(Data):
    """PyG ``Data`` subclass for surface meshes.

    Overrides concatenation behavior so that the ``y`` field is stacked along a
    new dimension (used to hold spherical maps) instead of being concatenated,
    and reports a length of 1 (a single graph per object).
    """
    def __cat_dim__(self, key, value, *args, **kwargs):
        #along y we want a new dimension to handle the spherical maps
        if key == 'y':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
    def __len__(self):
        return 1

class LoadedDataset(Dataset):
    """Dataset that loads surface-mesh graphs from a list of ``.pt`` files.

    Args:
        filenames: list of paths to ``.pt`` files, each holding one graph.
        transform: if True, apply a random rigid transformation on load.
        translate: if True, also apply a random translation in the transform.
    """
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
        """Load a single graph from a ``.pt`` file path."""
        if type(filename) == str:
            data = torch.load(filename, weights_only=False)
        return data
    
    def get_MaxDist(self):
        """Return the maximum vertex norm (distance to origin) across the dataset."""
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
        """Apply a random rotation (and optional translation) to a graph's positions."""
        # Apply a random rigid transformation to the data
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
    """Dataset loader for the COMA database.

    Loads one or two parallel lists of ``.pt`` files. When both lists have the
    same length the items are returned as pairs. Optionally normalizes features
    using precomputed statistics and appends a dummy node.

    Args:
        filenames1: primary list of ``.pt`` file paths.
        filenames2: optional second list, returned paired with ``filenames1``.
        normalize: if True, standardize features using stored mean/std tensors.
        dummy_node: if True, append an extra zero-valued node to each graph.
    """
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
            # The two input lists differ in length; only the first input will be processed.
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
    """Split a dataset into train/validation/test sets and build dataloaders.

    The split fractions are ``train_set_percentage``, ``val_set_percentage`` and
    the remainder for the test set, using a fixed random seed for reproducibility.

    Args:
        dataset: dataset instance to split.
        batch_size: batch size for the training loader.
        batchtest: batch size for the validation and test loaders.
        train_set_percentage: fraction of samples used for training.
        val_set_percentage: fraction of samples used for validation.
        shuffle: whether to shuffle the training loader.
        num_workers: number of dataloader workers.
        pin_memory: whether to pin memory in the dataloaders.

    Returns:
        tuple of (train_loader, val_loader, test_loader).
    """
    train_data, val_data, test_data = torch.utils.data.random_split(dataset,[train_set_percentage,val_set_percentage,1-(train_set_percentage+val_set_percentage)], torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, shuffle=False,  batch_size=batchtest, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, shuffle=False,batch_size=batchtest, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader