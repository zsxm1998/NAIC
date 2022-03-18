import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import six
import faiss
import math

DIM_NUM = 128
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


def abnormal_compress(compressed_query_fea_dir, vectors, basenames, faiss_index, bytes_rate):
    # bytes_rate == 64
    block_num, each_block_len = 32, 16
    if bytes_rate == 128:
        block_num, each_block_len = 64, 16
    elif bytes_rate == 256:
        block_num, each_block_len = 128, 16
    for vector, bname in zip(vectors, basenames):
        compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
        code = ''
        for i in range(bytes_rate * 8 - block_num * each_block_len):
            code = code + '1'
        # bytes_rate=64: 前面都是1，后面存信息
        ra = np.array(vector).reshape(block_num, -1).astype('float32')
        k = 1
        D, I = faiss_index.search(ra, k)
        for i in I:
            min_idx = i[0]
            # 2 ** 16 = 65536 = dictionary size
            for j in range(each_block_len):
                if min_idx & 1:
                    code = code + '1'
                else:
                    code = code + '0'
                min_idx = min_idx // 2
        with open(compressed_fea_path, 'wb') as f:
            for i in range(len(code) // 8):
                out = 0
                for j in range(i * 8, (i + 1) * 8):
                    out = out * 2
                    if code[j] == '1':
                        out = out + 1
                f.write(six.int2byte(out))


def normal_compress(compressed_query_fea_dir, vector, basename):
    compressed = vector.half().numpy().astype('<f2')
    
    for b, bname in enumerate(basename):
        compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
        compressed[b].tofile(compressed_fea_path)

# fast find the nearest number
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

# basenames, vectors: list with is_normal cases
def huffman_compress(compressed_query_fea_dir, vectors, basenames, forward, nums, bytes_rate):
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

    featuredataset = FeatureDataset(query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    normal = len(featuredataset) != 30000

    # normal cases
    if normal:
        if bytes_rate != 256:
            huffman_dict = torch.load(os.path.join(root, f'project/huffman_integral.pth'))
            forward = huffman_dict[f'for{bytes_rate}']
            nums = huffman_dict[f'nums{bytes_rate}']
            for vector, basename in featureloader:
                vector = vector[:, :DIM_NUM]
                huffman_compress(compressed_query_fea_dir, vector, basename, forward, nums, bytes_rate)
        else:
            for vector, basename in featureloader:
                vector = vector[:, :DIM_NUM]
                normal_compress(compressed_query_fea_dir, vector, basename)

    # abnormal cases
    if not normal:
        # load code dictionary
        code_dict = np.fromfile(os.path.join(root, f'project/compress_dict_bd_{bytes_rate}.dict'), dtype='<f4')
        shape_d = {64: 64, 128: 32, 256: 16}[bytes_rate]
        code_dict = code_dict.reshape(-1, shape_d)
        cpu_index = faiss.IndexFlatL2(shape_d)
        cpu_index.add(code_dict)
        for vector, basename in featureloader:
            abnormal_compress(compressed_query_fea_dir, vector, basename, cpu_index, bytes_rate)

    print('Compression Done')
