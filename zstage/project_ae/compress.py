import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import six
import faiss

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


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, input_dim)
        self.el2 = nn.Linear(input_dim, intermediate_dim)
        self.en2 = nn.BatchNorm1d(intermediate_dim)

    def forward(self, x):
        return self.en2(self.el2(self.relu(self.el1(x))))


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


def normal_compress(compressed_query_fea_dir, encoder, device, normal_cases, normal_basenames, bytes_rate):
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(32, DIM_NUM)
    encoder.load_state_dict(torch.load(os.path.join(root, f'project/Encoder_32_best.pth')))
    encoder.to(device)
    encoder.eval()

    # load code dictionary
    code_dict = np.fromfile(os.path.join(root, f'project/compress_65536x2048.dict'), dtype='<f4')
    shape_d = {64: 128, 128: 64, 256: 32}[bytes_rate]
    code_dict = code_dict.reshape(-1, shape_d)
    cpu_index = faiss.read_index(os.path.join(root, f'project/faiss_index_br_{bytes_rate}.index'))
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(code_dict)

    featuredataset = FeatureDataset(query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    normal_list, normal_basenames_list = [], []
    abnormal_list, abnormal_basenames_list = [], []
    for vector, basename in featureloader:
        basename = np.array(basename)
        normal_res = vector[:, DIM_NUM:].eq(0).all(dim=-1)
        normal_list.append(vector[normal_res==True, :DIM_NUM])
        normal_basenames_list.append(basename[normal_res==True])
        abnormal_list.append(vector[normal_res==False])
        abnormal_basenames_list.append(basename[normal_res==False])

    normal_cases = torch.cat(normal_list, dim=0)
    normal_basenames = np.concatenate(normal_basenames_list, axis=0)
    abnormal_cases = torch.cat(abnormal_list, dim=0)
    abnormal_basenames = np.concatenate(abnormal_basenames_list, axis=0)
    del normal_list, normal_basenames_list, abnormal_list, abnormal_basenames_list

    # normal cases
    normal_compress(compressed_query_fea_dir, encoder, device, normal_cases, normal_basenames, bytes_rate)
    # abnormal cases
    abnormal_compress(compressed_query_fea_dir, abnormal_cases, abnormal_basenames, gpu_index, bytes_rate)

    print('Compression Done')
