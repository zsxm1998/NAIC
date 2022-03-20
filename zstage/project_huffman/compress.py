import math
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import six

DIM_NUM = 180
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

# fast find the nearest number
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


def huffman_compress(root, compressed_query_fea_dir, bytes_rate, featureloader):
    huffman_dict = torch.load(os.path.join(root, f'project/huffman_180_len_len_len.pth'))
    forward = huffman_dict[f'for{bytes_rate}']
    nums = huffman_dict[f'nums{bytes_rate}']
    for vectors, basenames in featureloader:
        vectors = vectors[:, :DIM_NUM]
        for vector, bname in zip(vectors, basenames):
            compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
            output_len = 0
            code = ''
            with open(compressed_fea_path, 'wb') as f:
                for extract_num in vector:
                    extract_num = float(extract_num)
                    closest_num = find_nearest(nums, extract_num)
                    code = code + forward[closest_num]
                out = 0
                while len(code) > 8 and output_len < bytes_rate:
                    for x in range(8):
                        out = out << 1
                        if code[x] == '1':
                            out = out | 1
                    code = code[8:]
                    f.write(six.int2byte(out))
                    output_len += 1
                    out = 0
                # 处理剩下来的不满8位的code
                out = 0
                for i in range(len(code)):
                    out = out << 1
                    if code[i]=='1':
                        out = out | 1
                for i in range(8 - len(code)):
                    out = out << 1
                # 把最后一位给写入到文件当中
                if output_len < bytes_rate:
                    f.write(six.int2byte(out))
                    output_len += 1
                # 补成一样长度
                while output_len < bytes_rate:
                    f.write(six.int2byte(0))
                    output_len += 1
                    

@torch.no_grad()
def compress(bytes_rate, root=''):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = os.path.join(root, 'query_feature')
    compressed_query_fea_dir = os.path.join(root, 'compressed_query_feature/{}'.format(bytes_rate))
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    # set to True, use huffman
    featuredataset = FeatureDataset(query_fea_dir)
    normal = featuredataset.normal
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE if normal else ABNORMAL_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # normal cases
    if normal:
        huffman_compress(root, compressed_query_fea_dir, bytes_rate, featureloader)

    # abnormal cases
    if not normal:
        abnormal_compress(root, compressed_query_fea_dir, bytes_rate, featureloader)

    print('Compression Done')
