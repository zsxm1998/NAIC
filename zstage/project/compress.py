import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, input_dim)
        self.el2 = nn.Linear(input_dim, intermediate_dim)

    def forward(self, x):
        return F.normalize((self.el2(self.relu(self.el1(x)))), dim=1)


@torch.no_grad()
def compress(bytes_rate):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = 'query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    net = Encoder(32, 2048)
    net.load_state_dict(torch.load(f'project/Encoder_32_best.pth', map_location=torch.device('cpu')))
    net.eval()

    featuredataset = FeatureDataset(query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        compressed = net(vector)
        if bytes_rate == 64:
            compressed = compressed.half()
        elif bytes_rate == 256:
            compressed = compressed.double()
        for i, bname in enumerate(basename):
            compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
            with open(compressed_fea_path, 'wb') as bf:
                bf.write(compressed[i].numpy().tobytes())

    print('Compression Done')
