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
    # bytes_rate == 64
    block_num, each_block_len = 16, 20
    if bytes_rate == 128:
        block_num, each_block_len = 32, 20
    elif bytes_rate == 256:
        block_num, each_block_len = 64, 22

    for vector, bname in zip(vectors, basenames):
        compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
        code = ''
        for i in range(bytes_rate * 8 - block_num * each_block_len):
            code = code + '1'
        # bytes_rate=64: 前面都是1，后面存信息
        ra = np.array(vector).reshape(block_num, -1).astype('float32')
        k = 1
        D, I = gpu_index.search(ra, k)
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


@torch.no_grad()
def normal_compress(root, compressed_query_fea_dir, bytes_rate, featureloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ae_net = AEModel('efficientnet_b4(num_classes={})', extractor_out_dim=DIM_NUM, compress_dim=32)
    ae_net.load_param(os.path.join(root, f'project/Net_best.pth'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ae_net.to(device)
    ae_net.eval()
    for vector, basename in featureloader:
        vector = vector[:, :DIM_NUM].to(device)
        compressed = ae_net.compress()
    
    
    
    for i in range(0, len(normal_cases), BATCH_SIZE):
        j = min(i+BATCH_SIZE, len(normal_cases))
        vector = normal_cases[i: j]
        basename = normal_basenames[i: j]
        if bytes_rate != 256:
            vector = vector.to(device)
            compressed = encoder(vector).cpu()
            if bytes_rate == 64:
                compressed = compressed.half().numpy().astype('<f2')
            else:
                compressed = compressed.numpy().astype('<f4')
        else:
            compressed = vector.half().numpy().astype('<f2')
        
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
        pass

    # abnormal cases
    if not normal:
        pass

    print('Compression Done')
