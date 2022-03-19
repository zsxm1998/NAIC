import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import six
from .ae import AEModel

DIM_NUM = 1024
BATCH_SIZE = 512

class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


def abnormal_compress(compressed_query_fea_dir, vectors, basenames, gpu_index, bytes_rate):
    pass


@torch.no_grad()
def normal_compress(root, compressed_query_fea_dir, bytes_rate, featureloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ae_net = AEModel('efficientnet_b4(num_classes={})', extractor_out_dim=DIM_NUM, compress_dim=32)
    ae_net.load_param(os.path.join(root, f'project/Net_best.pth'))
    ae_net.to(device)
    ae_net.eval()
    for vector, basename in featureloader:
        vector = vector[:, :DIM_NUM].to(device)
        compressed = ae_net.compress(vector, bytes_rate).cpu().half().numpy().astype('<f2')
        for b, bname in enumerate(basename):
            compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
            compressed[b].tofile(compressed_fea_path)


@torch.no_grad()
def compress(bytes_rate, root=''):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = os.path.join(root, 'query_feature')
    compressed_query_fea_dir = os.path.join(root, 'compressed_query_feature/{}'.format(bytes_rate))
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    featuredataset = FeatureDataset(query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    normal = len(featuredataset) != 30000

    # normal cases
    if normal:
        normal_compress(root, compressed_query_fea_dir, bytes_rate, featureloader)

    # abnormal cases
    if not normal:
        pass

    print('Compression Done')
