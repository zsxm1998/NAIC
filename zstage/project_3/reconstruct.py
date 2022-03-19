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
ABNORMAL_BATCH_SIZE = 64
NORMAL_COUNT = 30000


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    #assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))
        self.normal = len(self.query_fea_paths) != NORMAL_COUNT

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        if self.normal:
            vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f2')).float()
        else:
            vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype=np.int16))
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


def abnormal_reconstruct(root, reconstructed_query_fea_dir, bytes_rate, featureloader):
    # load code dictionary
    block_size = 2048 // (bytes_rate // 2)
    codebook = torch.from_numpy(np.load(os.path.join(root, f'project/gen_final_big_65400x{block_size}.npy')))
    for vector, basename in featureloader:
        for i, bname in enumerate(basename):
            compessed_fea = vector[i].int()
            reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
            idx = compessed_fea + 32768 # int16 -> int
            reconstructed_fea = np.zeros(2048, dtype=np.float32)
            for j in range(0, 2048, block_size):
                reconstructed_fea[j:j + block_size] = codebook[idx[j // block_size], :].numpy()
            write_feature_file(reconstructed_fea, reconstructed_fea_path)


@torch.no_grad()
def normal_reconstruct(root, reconstructed_query_fea_dir, bytes_rate, featureloader):
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

    featuredataset = FeatureDataset(compressed_query_fea_dir)
    normal = featuredataset.normal
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE if normal else ABNORMAL_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # normal cases
    if normal:
        normal_reconstruct(root, reconstructed_query_fea_dir, bytes_rate, featureloader)

    # abnormal cases
    if not normal:
        abnormal_reconstruct(root, reconstructed_query_fea_dir, bytes_rate, featureloader)

    print('Reconstruction Done')
