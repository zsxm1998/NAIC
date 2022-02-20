import os
import glob
import zipfile
import numpy as np
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def extract_zipfile(dir_input: str, dir_dest: str):
    files = zipfile.ZipFile(dir_input, "r")
    for file in files.namelist():
        if file.find("__MACOSX")>=0 or file.startswith('.'): continue
        else:
            files.extract(file, dir_dest)
    files.close()
    return 1

class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim=463):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, 512)
        self.en1 = nn.LayerNorm(512)
        self.el2 = nn.Linear(512, 256)
        self.en2 = nn.LayerNorm(256)
        self.el3 = nn.Linear(256, intermediate_dim)
        self.en3 = nn.LayerNorm(intermediate_dim)

    def forward(self, x):
        x = self.relu(self.en1(self.el1(x)))
        x = self.relu(self.en2(self.el2(x)))
        out = self.relu(self.en3(self.el3(x)))
        return out

class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))
        self.no_zero_dim = torch.load('./not_zero_dim.pt')

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))[self.no_zero_dim]
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename

@torch.no_grad()
def compress_all(input_path: str, bytes_rate: int):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    net = Encoder(int(bytes_rate/4))
    net.load_state_dict(torch.load(f'./Encoder_{bytes_rate}.pth', map_location=torch.device('cpu')))
    net.eval()

    featuredataset = FeatureDataset(input_path)
    featureloader = DataLoader(featuredataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        compressed = net(vector)
        for i, bname in enumerate(basename):
            compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
            with open(compressed_fea_path, 'wb') as bf:
                bf.write(compressed[i].numpy().tobytes())

    print('Encode Done for bytes_rate' + str(bytes_rate))


def compress(test_path:str, byte: str):
    query_fea_dir = 'query_feature'
    extract_zipfile(test_path, query_fea_dir)
    compress_all(query_fea_dir, int(byte))
    shutil.rmtree(query_fea_dir)
    return 1
