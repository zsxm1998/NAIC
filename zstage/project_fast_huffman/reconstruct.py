import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


@torch.no_grad()
def reconstruct(bytes_rate, root=''):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    compressed_query_fea_dir = os.path.join(root, 'compressed_query_feature/{}'.format(bytes_rate))
    reconstructed_query_fea_dir = os.path.join(root, 'reconstructed_query_feature/{}'.format(bytes_rate))
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    # load huffman
    state = torch.load(os.path.join(root, f'project/huffman.pth'))
    for64 = state['for64']
    rev64 = state['rev64']
    nums64 = state['nums64']
    for128 = state['for128']
    rev128 = state['rev128']
    nums128 = state['nums128']

    featuredataset = FeatureDataset(compressed_query_fea_dir, bytes_rate=bytes_rate)
    featureloader = DataLoader(featuredataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    for vector, basename in featureloader:
        compressed_fea_path = os.path.join(compressed_query_fea_dir, basename[0] + '.dat')
        if bytes_rate != 256:
            bname = basename[0]
            with open(compressed_fea_path, 'rb') as f:
                feature_len = 2048
                fea = np.zeros(feature_len, dtype='<f4')
                filedata = f.read()
                filesize = f.tell()
            reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
            if bytes_rate == 64:
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
                        if code in rev64:
                            fea[idx] = rev64[code]
                            idx = idx + 1
                            code = ''
                fea.tofile(reconstructed_fea_path)
            elif bytes_rate == 128:
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
                        if code in rev128:
                            fea[idx] = rev128[code]
                            idx = idx + 1
                            code = ''
                fea.tofile(reconstructed_fea_path)
        else:
            reconstructed = vector
            expand_r = torch.zeros(reconstructed.shape[0], 2048, dtype=reconstructed.dtype)
            expand_r[:, :128] = reconstructed
            expand_r = expand_r.numpy().astype('<f4')

            for i, bname in enumerate(basename):
                reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
                # with open(reconstructed_fea_path, 'wb') as bf:
                #     bf.write(expand_r[i].numpy().tobytes())
                expand_r[i].tofile(reconstructed_fea_path)

    print('Reconstruction Done')
