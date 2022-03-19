import imp
import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .ae import AEModel

DIM_NUM = 1024
BATCH_SIZE = 512


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    #assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


class FeatureDataset(Dataset):
    def __init__(self, file_dir, data_type=float):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))
        self.data_type = data_type

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        if self.data_type == float:
            vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f2')).float()
        elif self.data_type == int:
            vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<i2'))
        else:
            raise ValueError(f'data_type wrong:{self.data_type}')
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


def abnormal_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, basenames, code_dict, bytes_rate):
    pass


@torch.no_grad()
def normal_reconstruct(root, compressed_query_fea_dir, reconstructed_query_fea_dir, bytes_rate):
    featuredataset = FeatureDataset(compressed_query_fea_dir, float)
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ae_net = AEModel('efficientnet_b4(num_classes={})', extractor_out_dim=DIM_NUM, compress_dim=32)
    ae_net.load_param(os.path.join(root, f'project/Net_best.pth'))
    ae_net.to(device)
    ae_net.eval()
    for vector, basename in featureloader:
        vector = vector.to(device)
        reconstructed = ae_net.reconstract(vector, bytes_rate).cpu()
        expand_r = torch.zeros(reconstructed.shape[0], 2048, dtype=reconstructed.dtype)
        expand_r[:, :DIM_NUM] = reconstructed
        expand_r = expand_r.numpy().astype('<f4')
        for b, bname in enumerate(basename):
            reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
            expand_r[b].tofile(reconstructed_fea_path)


@torch.no_grad()
def reconstruct(bytes_rate, root=''):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    compressed_query_fea_dir = os.path.join(root, 'compressed_query_feature/{}'.format(bytes_rate))
    reconstructed_query_fea_dir = os.path.join(root, 'reconstructed_query_feature/{}'.format(bytes_rate))
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    normal = len(glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))) != 30000

    # normal cases
    if normal:
        normal_reconstruct(root, compressed_query_fea_dir, reconstructed_query_fea_dir, bytes_rate)

    # abnormal cases
    if not normal:
        pass

    print('Reconstruction Done')
