import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import six


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


@torch.no_grad()
def compress(bytes_rate, root=''):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = os.path.join(root, 'query_feature')
    compressed_query_fea_dir = os.path.join(root, 'compressed_query_feature/{}'.format(bytes_rate))
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    # load huffman
    state = torch.load(os.path.join(root, f'project/huffman.pth'))
    for64 = state['for64']
    rev64 = state['rev64']
    nums64 = state['nums64']
    for128 = state['for128']
    rev128 = state['rev128']
    nums128 = state['nums128']

    featuredataset = FeatureDataset(query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        if bytes_rate != 256:
            vector = vector[:, :DIM_NUM]
            if bytes_rate == 64:
                bname = basename[0]
                compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
                output_len = 0
                code = ''
                with open(compressed_fea_path, 'wb') as f:
                    for extract_num in vector[0]:
                        extract_num = float(extract_num)
                        distance = 10000000 # inf
                        closest_num = -1
                        for num in nums64:
                            if abs(num - extract_num) < distance:
                                closest_num = num
                                distance = abs(num - extract_num)
                        code = code + for64[closest_num]
                    out = 0
                    while len(code) > 8 and output_len < 64:
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
                    if output_len < 64:
                        f.write(six.int2byte(out))
            elif bytes_rate == 128:
                bname = basename[0]
                compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
                output_len = 0
                code = ''
                with open(compressed_fea_path, 'wb') as f:
                    for extract_num in vector[0]:
                        extract_num = float(extract_num)
                        distance = 10000000 # inf
                        closest_num = -1
                        for num in nums128:
                            if abs(num - extract_num) < distance:
                                closest_num = num
                                distance = abs(num - extract_num)
                        code = code + for128[closest_num]
                    out = 0
                    while len(code) > 8 and output_len < 128:
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
                    if output_len < 128:
                        f.write(six.int2byte(out))
        else:
            compressed = vector[:, :DIM_NUM].half().numpy().astype('<f2')
            for i, bname in enumerate(basename):
                compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
                # with open(compressed_fea_path, 'wb') as bf:
                #     bf.write(compressed[i].numpy().tobytes())
                compressed[i].tofile(compressed_fea_path)

    print('Compression Done')
