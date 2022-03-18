import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DIM_NUM = 128
BATCH_SIZE = 512


class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))[:DIM_NUM]
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
        self.en2 = nn.BatchNorm1d(intermediate_dim)

    def forward(self, x):
        return self.en2(self.el2(self.relu(self.el1(x))))


@torch.no_grad()
def compress(bytes_rate, root=''):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = os.path.join(root, 'query_feature')
    compressed_query_fea_dir = os.path.join(root, 'compressed_query_feature/{}'.format(bytes_rate))
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(32, DIM_NUM)
    encoder.load_state_dict(torch.load(os.path.join(root, f'project/Encoder_32_best.pth')))
    encoder.to(device)
    encoder.eval()

    featuredataset = FeatureDataset(query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        if bytes_rate != 256:
            vector = vector.to(device)
            compressed = encoder(vector).cpu()
            if bytes_rate == 64:
                compressed = compressed.half().numpy().astype('<f2')
            else:
                compressed = compressed.numpy().astype('<f4')
        else:
            compressed = vector.half().numpy().astype('<f2')
        
        for i, bname in enumerate(basename):
            compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
            # with open(compressed_fea_path, 'wb') as bf:
            #     bf.write(compressed[i].numpy().tobytes())
            compressed[i].tofile(compressed_fea_path)

    print('Compression Done')
