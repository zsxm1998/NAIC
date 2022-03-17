import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import six
import faiss


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


@torch.no_grad()
def compress(bytes_rate, root=''):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = os.path.join(root, 'query_feature')
    compressed_query_fea_dir = os.path.join(root, 'compressed_query_feature/{}'.format(bytes_rate))
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(32, 128)
    encoder.load_state_dict(torch.load(os.path.join(root, f'project/Encoder_32_best.pth')))
    encoder.to(device)
    encoder.eval()

    featuredataset = FeatureDataset(query_fea_dir)
    featureloader = DataLoader(featuredataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # load code dictionary
    code_dict = np.fromfile(os.path.join(root, f'project/compress.dict'), dtype='<f4')
    code_dict = code_dict.reshape(-1, 128)
    index = faiss.IndexFlatL2(128)
    index.add(code_dict)

    # load huffman
    state = torch.load(os.path.join(root, f'project/huffman.pth'))
    for64 = state['for64']
    rev64 = state['rev64']
    nums64 = state['nums64']
    for128 = state['for128']
    rev128 = state['rev128']
    nums128 = state['nums128']

    normal_list, normal_basenames = [], []
    abnormal_list, abnormal_basenames = [], []
    for vector, basename in featureloader:
        basename = np.array(basename)
        normal_res = vector[:, 128:].eq(0).all(dim=-1)
        normal_list.append(vector[normal_res==True])
        normal_basenames.append(basename[normal_res==True])
        abnormal_list.append(vector[normal_res==False])
        abnormal_basenames.append(basename[normal_res==False])

    normal_list = torch.cat(normal_list, dim=0)
    normal_basenames = np.concatenate(normal_basenames, axis=0)
    abnormal_list = torch.cat(abnormal_list, dim=0)
    abnormal_basenames = np.concatenate(abnormal_basenames, axis=0)
    

    # for vector, basename in featureloader:
    #     # is_normal 
    #     is_normal = True
    #     for i in range(128, 2048):
    #         if float(vector[0][i]) != 0.00000 :
    #             is_normal = False
    #             break
    #     if is_normal:
    #         # 正常情况
    #         if bytes_rate != 256:
    #             vector = vector[:, :128]
    #             if bytes_rate == 64:
    #                 bname = basename[0]
    #                 compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
    #                 output_len = 0
    #                 code = ''
    #                 with open(compressed_fea_path, 'wb') as f:
    #                     for extract_num in vector[0]:
    #                         extract_num = float(extract_num)
    #                         distance = 10000000 # inf
    #                         closest_num = -1
    #                         for num in nums64:
    #                             if abs(num - extract_num) < distance:
    #                                 closest_num = num
    #                                 distance = abs(num - extract_num)
    #                         code = code + for64[closest_num]
    #                     out = 0
    #                     while len(code) > 8 and output_len < 64:
    #                         for x in range(8):
    #                             out = out << 1
    #                             if code[x] == '1':
    #                                 out = out | 1
    #                         code = code[8:]
    #                         f.write(six.int2byte(out))
    #                         output_len += 1
    #                         out = 0
    #                     # 处理剩下来的不满8位的code
    #                     out = 0
    #                     for i in range(len(code)):
    #                         out = out << 1
    #                         if code[i]=='1':
    #                             out = out | 1
    #                     for i in range(8 - len(code)):
    #                         out = out << 1
    #                     # 把最后一位给写入到文件当中
    #                     if output_len < 64:
    #                         f.write(six.int2byte(out))
    #             elif bytes_rate == 128:
    #                 bname = basename[0]
    #                 compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
    #                 output_len = 0
    #                 code = ''
    #                 with open(compressed_fea_path, 'wb') as f:
    #                     for extract_num in vector[0]:
    #                         extract_num = float(extract_num)
    #                         distance = 10000000 # inf
    #                         closest_num = -1
    #                         for num in nums128:
    #                             if abs(num - extract_num) < distance:
    #                                 closest_num = num
    #                                 distance = abs(num - extract_num)
    #                         code = code + for128[closest_num]
    #                     out = 0
    #                     while len(code) > 8 and output_len < 128:
    #                         for x in range(8):
    #                             out = out << 1
    #                             if code[x] == '1':
    #                                 out = out | 1
    #                         code = code[8:]
    #                         f.write(six.int2byte(out))
    #                         output_len += 1
    #                         out = 0
    #                     # 处理剩下来的不满8位的code
    #                     out = 0
    #                     for i in range(len(code)):
    #                         out = out << 1
    #                         if code[i]=='1':
    #                             out = out | 1
    #                     for i in range(8 - len(code)):
    #                         out = out << 1
    #                     # 把最后一位给写入到文件当中
    #                     if output_len < 128:
    #                         f.write(six.int2byte(out))
    #         else:
    #             compressed = vector[:, :128].half().numpy().astype('<f2')
    #             for i, bname in enumerate(basename):
    #                 compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
    #                 # with open(compressed_fea_path, 'wb') as bf:
    #                 #     bf.write(compressed[i].numpy().tobytes())
    #                 compressed[i].tofile(compressed_fea_path)
    #     else:
    #         # 其他情况
    #         vector = vector.numpy().tolist()[0]
    #         for i, bname in enumerate(basename):
    #             compressed_fea_path = os.path.join(compressed_query_fea_dir, bname + '.dat')
    #             with open(compressed_fea_path, 'wb') as f:
    #                 # 16 * 16 = 256
    #                 for i in range(0, 16):
    #                     ra = np.array(vector[i * 128: (i + 1) * 128]).reshape(1, -1).astype('float32')
    #                     k = 1
    #                     D, I = index.search(ra, k)
    #                     min_idx = int(I[0])
    #                     for j in range(2):
    #                         f.write(six.int2byte(min_idx % 256))
    #                         min_idx = min_idx // 256

    # print('Compression Done')
