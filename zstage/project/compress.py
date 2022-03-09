import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))[:512]
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, 4096)
        self.en1 = nn.BatchNorm1d(4096)
        self.dp1 = nn.Dropout(p=0.5)
        self.el2 = nn.Linear(4096, 2048)
        self.en2 = nn.BatchNorm1d(2048)
        self.dp2 = nn.Dropout(p=0.3)
        self.el3 = nn.Linear(2048, 1024)
        self.en3 = nn.BatchNorm1d(1024)
        self.el4 = nn.Linear(1024, 512)
        self.en4 = nn.BatchNorm1d(512)
        self.el5 = nn.Linear(512, intermediate_dim)
        self.en5 = nn.BatchNorm1d(intermediate_dim)

    def forward(self, x):
        x = self.dp1(self.relu(self.en1(self.el1(x))))
        x = self.dp2(self.relu(self.en2(self.el2(x))))
        x = self.relu(self.en3(self.el3(x)))
        x = self.relu(self.en4(self.el4(x)))
        out = self.en5(self.el5(x))
        return out


@torch.no_grad()
def compress(bytes_rate):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = 'query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    net = Encoder(int(bytes_rate/2), 512)
    net.load_state_dict(torch.load(f'project/Encoder_{bytes_rate}.pth', map_location=torch.device('cpu')))
    net.eval()

    featuredataset = FeatureDataset(query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        compressed = net(vector).half()
        for i, bname in enumerate(basename):
            compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
            with open(compressed_fea_path, 'wb') as bf:
                bf.write(compressed[i].numpy().tobytes())

    print('Compression Done')
