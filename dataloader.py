"""Loads the game line up data with game result.
"""

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import time


class Dataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, labels_data, indices=None, model_type=None):
        """Set the path for the dataset.
        
        Args:
            dataset: annotation hdf5 location.
        """
        self.dataset = dataset
        self.labels_data = labels_data
        self.indices = indices
        self.model_type = model_type

    def __getitem__(self, index):
        """Returns features and labels.
        """
        if not hasattr(self, 'questions'):
            annos = h5py.File(self.dataset, 'r')
            #self.features = annos['features']
            if self.model_type == 'mlp':
                self.eegs = annos['mlp']#self.features[:, 11:]
            else:
                self.eegs = annos['no-mlp']#self.features[:, 11:].reshape(10, -1) 
            self.conditions = annos['conditions']#self.features[:, :11]
            self.labels = pd.read_csv(self.labels_data).values[:, 1]

        if self.indices is not None:
            index = self.indices[index]

        conditions = self.conditions[index]
        eeg = self.eegs[index]    
        label = self.labels[index]
        conditions = torch.from_numpy(np.array(conditions))
        eeg = torch.from_numpy(np.array(eeg))
        label = torch.from_numpy(np.array(label))
        return conditions, eeg, label

    def __len__(self):
        if self.indices is not None:
            return len(self.indices) 
        annos = h5py.File(self.dataset, 'r')
        return annos['features'].shape[0]

def collate_fn(data):
    conditions, eeg, label = zip(*data)
    conditions = torch.stack(conditions, 0).float()
    eegs = torch.stack(eeg, 0).float()
    labels = torch.stack(label, 0).long()
    return conditions, eegs, labels

def get_data_loader(dataset, labels_data, batch_size, shuffle, 
                    num_workers, model_type, indices=None):
    """Returns torch.utils.data.DataLoader for the dataset.
    """
    lm = Dataset(dataset, labels_data, indices=indices, model_type=model_type)
    data_loader = torch.utils.data.DataLoader(dataset=lm,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader