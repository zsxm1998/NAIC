import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DIM_NUM = 180
BATCH_SIZE = 512


#----------------------------------------------------------------------------------------!
# ----------------------------------------
# AE
# ----------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, input_dim)
        self.el2 = nn.Linear(input_dim, hidden_dim)
        self.en2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        return self.en2(self.el2(self.relu(self.el1(x))))


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.relu = nn.LeakyReLU(inplace=True)
        self.dl1 = nn.Linear(hidden_dim, 256)
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


class AE_Decoder(nn.Module):
    def __init__(self, io_dim, hidden_dim):
        super(AE_Decoder, self).__init__()
        self.encoder = Encoder(io_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, io_dim)

    def forward(self, x_encode):
        # x_encode = self.encoder(x)
        # x_encode.half().float()
        x = self.decoder(x_encode)
        return x


# ----------------------------------------
# Dataset
# ----------------------------------------
class DatDataset(Dataset):
    def __init__(self, dat_dir, transform=None):
        self.transform = transform
        self.samples = []  # dat_path, dat_name
        for dat_name in os.listdir(dat_dir):
            dat_path = os.path.join(dat_dir, dat_name)
            item = dat_path, dat_name
            self.samples.append(item)

    def __getitem__(self, index):
        dat_path, dat_name = self.samples[index]
        dat = np.fromfile(dat_path, dtype='<f2')
        dat = torch.from_numpy(dat)
        dat = dat.float()

        if self.transform is not None:
            dat = self.transform(dat)

        return dat, dat_name

    def __len__(self):
        return len(self.samples)



# ----------------------------------------
# data loader
# ----------------------------------------
def load_data(data_dir):
    dataset = DatDataset(dat_dir=data_dir,
                         transform=None)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=True)
    return data_loader
#----------------------------------------------------------------------------------------!


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
        huffman_dict = torch.load(os.path.join(root, f'project/huffman_180_len_len_area.pth'))
        reverse = huffman_dict[f'rev{bytes_rate}']
        nums = huffman_dict[f'nums{bytes_rate}']
        for vector, basename in featureloader:
            huffman_reconstruct(compressed_query_fea_dir, reconstructed_query_fea_dir, basename, reverse, nums, bytes_rate)
    
    # abnormal case
    if not normal:
        # ----------------------------------------
        # basic configuration
        # ----------------------------------------
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = AE_Decoder(io_dim=2048, hidden_dim=bytes_rate//2)
        model.load_state_dict(torch.load(os.path.join(root, 'project/ae_{}.pth'.format(bytes_rate//2))))
        model.to(device)

        data_loader = load_data(compressed_query_fea_dir)

        # ----------------------------------------
        # test
        # ----------------------------------------
        model.eval()
        for b, samples in enumerate(data_loader):
            inputs, names = samples
            inputs = inputs.to(device)

            with torch.set_grad_enabled(False):
                reconstructed = model(inputs)

            dats = reconstructed.cpu().numpy()
            for i, dat in enumerate(dats):
                dat_path = os.path.join(reconstructed_query_fea_dir, names[i])
                dat.astype('<f4').tofile(dat_path)

    print('Reconstruction Done')
