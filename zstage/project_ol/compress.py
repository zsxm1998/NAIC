import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import six
from torch import optim
import math
from tqdm import tqdm

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


class AE(nn.Module):
    def __init__(self, io_dim, hidden_dim):
        super(AE, self).__init__()
        self.encoder = Encoder(io_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, io_dim)

    def forward(self, x):
        x_encode = self.encoder(x)
        x_encode.half().float()
        x = self.decoder(x_encode)
        return x_encode, x


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
        dat = np.fromfile(dat_path, dtype='<f4')
        dat = torch.from_numpy(dat)

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


class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename


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
        huffman_dict = torch.load(os.path.join(root, f'project/huffman_180_len_len_area.pth'))
        forward = huffman_dict[f'for{bytes_rate}']
        nums = huffman_dict[f'nums{bytes_rate}']
        for vector, basename in featureloader:
            vector = vector[:, :DIM_NUM]
            huffman_compress(compressed_query_fea_dir, vector, basename, forward, nums, bytes_rate)

    # abnormal cases
    if not normal:
        # ----------------------------------------
        # basic configuration
        # ----------------------------------------
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = AE(io_dim=2048, hidden_dim=bytes_rate//2)
        model.load_state_dict(torch.load(os.path.join(root, 'project/ae_{}.pth'.format(bytes_rate//2))))
        model.to(device)
        for p in model.decoder.parameters():
            p.requires_grad = False

        data_loader = load_data(query_fea_dir)

        # ----------------------------------------
        # train
        # ----------------------------------------
        model.train()

        epoch = 100
        criterion = nn.MSELoss()  # reduce = True size_average = True
        optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)

        best_loss = None
        best_model = None
        for e in range(epoch):
            running_loss = 0
            running_num = 0
            for i, samples in enumerate(tqdm(data_loader, leave=False)):
                inputs, _ = samples
                inputs = inputs.to(device)
                _, outputs = model(inputs)

                loss = criterion(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += (loss.item() * inputs.size(0))
                running_num += inputs.size(0)
            running_loss = running_loss / running_num
            scheduler.step()

            if best_loss is None or best_loss > running_loss:
                best_loss = running_loss
                best_model = model.state_dict()

            # print('Epoch', e, best_loss)

        model.load_state_dict(best_model)

        # ----------------------------------------
        # test
        # ----------------------------------------
        model.eval()
        for b, samples in enumerate(data_loader):
            inputs, names = samples
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):
                compressed, _ = model(inputs)

            compressed = compressed.half()
            dats = compressed.cpu().numpy()
            for i, dat in enumerate(dats):
                dat_path = os.path.join(compressed_query_fea_dir, names[i])
                dat.astype('<f2').tofile(dat_path)

    print('Compression Done')
