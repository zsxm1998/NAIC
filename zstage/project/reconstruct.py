import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    #assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f2')).float()
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU(inplace=True)
        self.dl1 = nn.Linear(intermediate_dim, 512)
        self.dn1 = nn.BatchNorm1d(512)
        self.dl2 = nn.Linear(512, 1024)
        self.dn2 = nn.BatchNorm1d(1024)
        self.dl3 = nn.Linear(1024, 2048)
        self.dn3 = nn.BatchNorm1d(2048)
        self.dl4 = nn.Linear(2048, 4096)
        self.dn4 = nn.BatchNorm1d(4096)
        self.dl5 = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        x = self.relu(self.dn3(self.dl3(x)))
        x = self.relu(self.dn4(self.dl4(x)))
        out = self.dl5(x)
        return out


@torch.no_grad()
def reconstruct(bytes_rate):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    net = Decoder(int(int(bytes_rate)/2), 512)
    net.load_state_dict(torch.load(f'project/Decoder_{bytes_rate}.pth', map_location=torch.device('cpu')))
    net.eval()

    featuredataset = FeatureDataset(compressed_query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        reconstructed = net(vector)
        expand_r = torch.zeros(reconstructed.shape[0], 2048, dtype=reconstructed.dtype)
        expand_r[:, :512] = reconstructed

        for i, bname in enumerate(basename):
            reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
            with open(reconstructed_fea_path, 'wb') as bf:
                bf.write(expand_r[i].numpy().tobytes())

    print('Reconstruction Done')
