import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, Dataset
from scipy.signal import decimate
import os

from .transforms import get_transform
from .utils import make_path, read_memmap

class SyntheticDataset(Dataset):
    def __init__(self, 
            file_path,
            input_type, 
            output_type,
            is_ees=False,
            indices_path=None,
            ratio=1,
            is_train=True,
            input_transform=None, 
            target_transform=None, 
            share_transform=None):

        mode = 'train' if is_train else 'valid'        
  
        input_path = make_path('%s.npy' % (input_type), directory=os.path.join(file_path, mode))
        output_path = make_path('%s.npy' % (output_type), directory=os.path.join(file_path, mode))
        meta_path = make_path('meta.npy', directory=os.path.join(file_path, mode))

        self.inputs = np.load(input_path)
        self.outputs = np.load(output_path)
        self.meta = np.load(meta_path)
        
        print('Inputs: ', self.inputs.shape, self.meta.shape)      

        self.input_transform = get_transform(input_transform) if input_transform is not None else None
        self.target_transform = get_transform(target_transform) if target_transform is not None else None
        self.share_transform = get_transform(share_transform) if share_transform is not None else None
        
        self.is_ess = is_ees
        self.n_samples = int(self.outputs.shape[0])
        if (indices_path is not None) and (ratio < 1):
                indices = torch.load(indices_path)
                self.indices = indices[:int(indices.size(0)*ratio)]
                self.n_samples = int(self.indices.shape[0])
        else:
            self.indices = None
                            

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        input = self.inputs[index]
        output = self.outputs[index]
        meta = self.meta[index]
        
        input = torch.from_numpy(input).float()
        output = torch.from_numpy(output).float()
        meta = torch.from_numpy(meta).float()
       
#         if self.is_ess:
#             ees = self.ees[index]
#             ees = torch.from_numpy(ees).float()
#             meta = torch.from_numpy(self.meta[index]).float()
     
        
#         if self.share_transform is not None:
#             if self.is_ess:
#                 input, output, ees = self.share_transform(input, output, ees)
#             else:
#                 input, output = self.share_transform(input, output)
        
#         print(input)
#         print(self.input_transform.__dict__)
#         if self.input_transform is not None:
#             print('Input transform ', index)
#             input = self.input_transform(input)
#         print(input)
            
#         if self.target_transform is not None:
#             output = self.target_transform(output)            
       
        if self.is_ess:
            return input, meta
        else:
            return input, meta
        #####
        
#         if self.is_ess:
#             return input, output, ees, meta
#         else:
#             return input, output


        
    def __len__(self):
        return self.n_samples
