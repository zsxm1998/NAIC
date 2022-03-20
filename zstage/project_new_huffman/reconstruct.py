import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DIM_NUM = 180
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


def huffman_reconstruct(root, compressed_query_fea_dir, reconstructed_query_fea_dir, bytes_rate, featureloader):
    if bytes_rate != 256:
        huffman_dict = torch.load(os.path.join(root, f'project/huffman_128_len_len_1.pth'))
        reverse = huffman_dict[f'rev{bytes_rate}']
        nums = huffman_dict[f'nums{bytes_rate}']
        for vector, basenames in featureloader:
            # huffman_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, basename, reverse, nums, bytes_rate)
            for bname in basenames:
                compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
                reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
                with open(compressed_fea_path, 'rb') as f:
                    feature_len = 2048
                    fea = np.zeros(feature_len, dtype='<f4')
                    filedata = f.read()
                    filesize = f.tell()
                idx = 0
                code = ''
                for x in range(0, filesize):
                    #python3
                    c = filedata[x]
                    for i in range(8):
                        if c & 128:
                            code = code + '1'
                        else:
                            code = code + '0'
                        c = c << 1
                        if code in reverse:
                            fea[idx] = reverse[code]
                            idx = idx + 1
                            code = ''
                fea.tofile(reconstructed_fea_path)
    else:
        for vector, basename in featureloader:
            reconstructed = vector
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
        huffman_reconstruct(root, compressed_query_fea_dir, reconstructed_query_fea_dir, bytes_rate, featureloader)

    # abnormal cases
    if not normal:
        abnormal_reconstruct(root, reconstructed_query_fea_dir, bytes_rate, featureloader)

    print('Reconstruction Done')
