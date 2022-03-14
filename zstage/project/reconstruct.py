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
    def __init__(self, file_dir, bytes_rate):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))
        if bytes_rate == 64:
            self.dtype_str = '<f2'
        elif bytes_rate == 128:
            self.dtype_str = '<f4'
        elif bytes_rate == 256:
            self.dtype_str = '<f2'
        else:
            raise NotImplementedError(f'reconstruct FeatureDataset bytes_rate error!{bytes_rate}{type(bytes_rate)}')

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype=self.dtype_str)).float()
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.LeakyReLU(inplace=True)
        self.dl1 = nn.Linear(intermediate_dim, 256)
        self.dn1 = nn.BatchNorm1d(256)
        self.dl2 = nn.Linear(256, 512)
        self.dn2 = nn.BatchNorm1d(512)
        self.dl3 = nn.Linear(512, 1024)
        self.dn3 = nn.BatchNorm1d(1024)
        self.dp3 = nn.Dropout(0.05)
        self.dl4 = nn.Linear(1024, 2048)
        self.dn4 = nn.BatchNorm1d(2048)
        self.dp4 = nn.Dropout(0.1)
        self.dl5 = nn.Linear(2048, 4096)
        self.dn5 = nn.BatchNorm1d(4096)
        self.dp5 = nn.Dropout(0.2)
        self.dl6 = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        x = self.dp3(self.relu(self.dn3(self.dl3(x))))
        x = self.dp4(self.relu(self.dn4(self.dl4(x))))
        x = self.dp5(self.relu(self.dn5(self.dl5(x))))
        return self.dl6(x)


@torch.no_grad()
def reconstruct(bytes_rate):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    decoder = Decoder(32, 128)
    decoder.load_state_dict(torch.load(f'project/Decoder_32_best.pth'))
    decoder.to(device)
    decoder.eval()

    featuredataset = FeatureDataset(compressed_query_fea_dir, bytes_rate=bytes_rate)
    featureloader = DataLoader(featuredataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        if bytes_rate != 256:
            vector = vector.to(device)
            reconstructed = decoder(vector).cpu()
        else:
            reconstructed = vector
        
        expand_r = torch.zeros(reconstructed.shape[0], 2048, dtype=reconstructed.dtype)
        expand_r[:, :128] = reconstructed
        expand_r = expand_r.numpy().astype('<f4')

        for i, bname in enumerate(basename):
            reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
            # with open(reconstructed_fea_path, 'wb') as bf:
            #     bf.write(expand_r[i].numpy().tobytes())
            expand_r[i].tofile(reconstructed_fea_path)

    print('Reconstruction Done')
