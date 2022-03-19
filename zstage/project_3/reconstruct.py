import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DIM_NUM = 1024
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


class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.LeakyReLU(inplace=True)
        self.dl1 = nn.Linear(intermediate_dim, 256)
        self.dn1 = nn.BatchNorm1d(256)
        self.dl2 = nn.Linear(256, 512)
        self.dn2 = nn.BatchNorm1d(512)
        self.dl3 = nn.Linear(512, 1024)
        self.dn3 = nn.BatchNorm1d(1024)
        self.dp3 = nn.Dropout(0.05)
        self.dl4 = nn.Linear(1024, 2048)
        self.dn4 = nn.BatchNorm1d(2048)
        self.dp4 = nn.Dropout(0.1)
        self.dl5 = nn.Linear(2048, 4096)
        self.dn5 = nn.BatchNorm1d(4096)
        self.dp5 = nn.Dropout(0.2)
        self.dl6 = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        x = self.dp3(self.relu(self.dn3(self.dl3(x))))
        x = self.dp4(self.relu(self.dn4(self.dl4(x))))
        x = self.dp5(self.relu(self.dn5(self.dl5(x))))
        return self.dl6(x)


def abnormal_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, basenames, code_dict, bytes_rate):
    # bytes_rate == 64
    block_num, each_block_len = 16, 20
    if bytes_rate == 128:
        block_num, each_block_len = 32, 20
    elif bytes_rate == 256:
        block_num, each_block_len = 64, 22

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


def normal_reconstruct(reconstructed_query_fea_dir, decoder, device, normal_cases, normal_basenames, bytes_rate):
    for i in range(0, len(normal_cases), BATCH_SIZE):
        j = min(i+BATCH_SIZE, len(normal_cases))
        vector = normal_cases[i: j]
        basename = normal_basenames[i: j]
        if bytes_rate != 256:
            vector = vector.to(device)
            reconstructed = decoder(vector).cpu()
        else:
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    decoder = Decoder(32, DIM_NUM)
    decoder.load_state_dict(torch.load(os.path.join(root, f'project/Decoder_32_best.pth')))
    decoder.to(device)
    decoder.eval()

    featuredataset = FeatureDataset(compressed_query_fea_dir, bytes_rate=bytes_rate)
    featureloader = DataLoader(featuredataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # load code dictionary
    code_dict = np.fromfile(os.path.join(root, f'project/compress_65536x2048.dict'), dtype='<f4')
    shape_d = {64: 128, 128: 64, 256: 32}[bytes_rate]
    code_dict = code_dict.reshape(-1, shape_d)

    normal_list, normal_basenames_list = [], []
    abnormal_list, abnormal_basenames_list = [], []
    for vector, basename in featureloader:
        basename = np.array(basename)
        normal_res = torch.isnan(vector[:, :2]).eq(True).all(dim=-1)
        normal_list.append(vector[normal_res==False])
        normal_basenames_list.append(basename[normal_res==False])
        abnormal_list.append(vector[normal_res==True])
        abnormal_basenames_list.append(basename[normal_res==True])

    normal_cases = torch.cat(normal_list, dim=0)
    normal_basenames = np.concatenate(normal_basenames_list, axis=0)
    abnormal_basenames = np.concatenate(abnormal_basenames_list, axis=0)
    del normal_list, normal_basenames_list, abnormal_list, abnormal_basenames_list

    # normal cases
    normal_reconstruct(reconstructed_query_fea_dir, decoder, device, normal_cases, normal_basenames, bytes_rate)
    # abnormal cases
    abnormal_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, abnormal_basenames, code_dict, bytes_rate)

    print('Reconstruction Done')
