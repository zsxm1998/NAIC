import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim=463):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU(inplace=True)
        self.dl1 = nn.Linear(intermediate_dim, 256)
        self.dn1 = nn.LayerNorm(256)
        self.dl2 = nn.Linear(256, 512)
        self.dn2 = nn.LayerNorm(512)
        self.dl3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        out = self.dl3(x)
        return out

class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename

@torch.no_grad()
def reconstruct(byte_rate: str):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(byte_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(byte_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    net = Decoder(int(int(byte_rate)/4))
    net.load_state_dict(torch.load(f'./Decoder_{byte_rate}.pth', map_location=torch.device('cpu')))
    net.eval()

    no_zero_dim = torch.load('./not_zero_dim.pt')

    featuredataset = FeatureDataset(compressed_query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        reconstructed = net(vector)

        expand_r = torch.zeros(reconstructed.shape[0], 2048, dtype=reconstructed.dtype)
        expand_r[:, no_zero_dim] = reconstructed

        for i, bname in enumerate(basename):
            reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
            with open(reconstructed_fea_path, 'wb') as bf:
                bf.write(expand_r[i].numpy().tobytes())

    print('Decode Done' + byte_rate)
