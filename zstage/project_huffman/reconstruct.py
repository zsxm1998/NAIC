import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DIM_NUM = 128
BATCH_SIZE = 512


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


def abnormal_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, basenames, code_dict, bytes_rate):
    # bytes_rate == 64
    block_num, each_block_len = 32, 16
    if bytes_rate == 128:
        block_num, each_block_len = 64, 16
    elif bytes_rate == 256:
        block_num, each_block_len = 128, 16

    for bname in basenames:
        compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
        with open(compressed_fea_path, 'rb') as f:
            filedata = f.read()
            filesize = f.tell()
        code = ''
        for i in range(filesize):
            out = filedata[i]
            for j in range(8):
                if out & (1<<7):
                    code = code + '1'
                else:
                    code = code + '0'
                out = out << 1
                
        ori = []
        for i in range(bytes_rate * 8 - block_num * each_block_len, bytes_rate * 8, each_block_len):
            idx = 0
            for j in range(0, each_block_len):
                if code[i + j] == '1':
                    idx = idx + 2 ** j
            ori += code_dict[idx].tolist()
        fea = np.array(ori)
        with open(reconstructed_fea_path, 'wb') as f:
            f.write(fea.astype('<f4').tostring())


def normal_reconstruct(reconstructed_query_fea_dir, vector, basename):
    reconstructed = vector
    expand_r = torch.zeros(reconstructed.shape[0], 2048, dtype=reconstructed.dtype)
    expand_r[:, :DIM_NUM] = reconstructed
    expand_r = expand_r.numpy().astype('<f4')

    for b, bname in enumerate(basename):
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
        expand_r[b].tofile(reconstructed_fea_path)

def huffman_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, basenames, reverse, nums, bytes_rate):
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


@torch.no_grad()
def reconstruct(bytes_rate, root=''):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    compressed_query_fea_dir = os.path.join(root, 'compressed_query_feature/{}'.format(bytes_rate))
    reconstructed_query_fea_dir = os.path.join(root, 'reconstructed_query_feature/{}'.format(bytes_rate))
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    featuredataset = FeatureDataset(compressed_query_fea_dir, bytes_rate=bytes_rate)
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    normal = len(featuredataset) != 30000

    # normal case
    if normal:
        if bytes_rate != 256:
            huffman_dict = torch.load(os.path.join(root, f'project/huffman_integral.pth'))
            reverse = huffman_dict[f'rev{bytes_rate}']
            nums = huffman_dict[f'nums{bytes_rate}']
            for vector, basename in featureloader:
                huffman_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, basename, reverse, nums, bytes_rate)
        else:
            for vector, basename in featureloader:
                normal_reconstruct(reconstructed_query_fea_dir, vector, basename)
    
    # abnormal case
    if not normal:
        # load code dictionary
        code_dict = np.fromfile(os.path.join(root, f'project/compress_dict_bd_{bytes_rate}.dict'), dtype='<f4')
        shape_d = {64: 64, 128: 32, 256: 16}[bytes_rate]
        code_dict = code_dict.reshape(-1, shape_d)
        for vector, basename in featureloader:
            abnormal_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, basename, code_dict, bytes_rate)

    print('Reconstruction Done')
