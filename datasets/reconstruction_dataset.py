import os
import copy
import random
from os.path import join
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

class ReconstructionDataset(Dataset):
    def __init__(self, dir, memory=True, not_zero=False):
        self.file_dir = join(dir, 'train_feature')
        self.memory = memory
        if not_zero and os.path.exists(join(dir, 'not_zero_dim.pt')):
            self.not_zero_dim = torch.load(join(dir, 'not_zero_dim.pt'))
        else:
            self.not_zero_dim = None
        self.datas = []
        for file in sorted(os.listdir(self.file_dir)):
            if memory:
                v = torch.from_numpy(np.fromfile(join(self.file_dir, file), dtype='<f4'))
                if self.not_zero_dim is not None:
                    v = v[self.not_zero_dim]
                self.datas.append(v)
            else:
                self.datas.append(file)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.memory:
            v = self.datas[index]
        else:
            v = torch.from_numpy(np.fromfile(join(self.file_dir, self.datas[index]), dtype='<f4'))
            if self.not_zero_dim is not None:
                v = v[self.not_zero_dim]
        return v