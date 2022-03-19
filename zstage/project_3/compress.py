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
ABNORMAL_BATCH_SIZE = 64
NORMAL_COUNT = 30000

class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))
        self.normal = len(self.query_fea_paths) != NORMAL_COUNT

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


def batch_dr(a, b):
    c = torch.pow(( a.unsqueeze(1) - b ), 2) # [a, b, d]
    d = torch.sum(c, dim=2)
    return d


def knn(query_features, codebook, block_size, device):
    nq, nd = query_features.shape
    best_idx = torch.zeros(nq, nd//block_size).to(torch.int16)

    loss_part = torch.zeros((nq, 1)).to(device)
    for j in range(0, nd, block_size):
        query_feature_block = query_features[:, j:j+block_size]
        dist = batch_dr(query_feature_block, codebook)
        val, ind = torch.topk(-dist, k=1, dim=1)
        best_idx[:, j // block_size] = ind[:, 0] - 32768 # int16 instead of uin16
    return best_idx


def abnormal_compress(root, compressed_query_fea_dir, bytes_rate, featureloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    block_size = 2048 // (bytes_rate // 2)
    codebook = torch.from_numpy(np.load(os.path.join(root, f'project/gen_final_big_65400x{block_size}.npy'))).to(device)
    for vector, basename in featureloader:
        vector = vector.to(device)
        idx = knn(query_features=vector, codebook=codebook, block_size=block_size, device=device)
        # write to file
        for i, bname in enumerate(basename):
            compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
            idx[i].cpu().numpy().astype(np.int16).tofile(compressed_fea_path)


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
    normal = featuredataset.normal
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE if normal else ABNORMAL_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # normal cases
    if normal:
        normal_compress(root, compressed_query_fea_dir, bytes_rate, featureloader)

    # abnormal cases
    if not normal:
        abnormal_compress(root, compressed_query_fea_dir, bytes_rate, featureloader)

    print('Compression Done')
