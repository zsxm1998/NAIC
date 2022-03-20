import os
import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F

from .ae import AEModel

DIM_NUM = 1024
BATCH_SIZE = 128


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(features, basenames, path):
    assert len(features) == len(basenames)
    for feature, basename in zip(features, basenames):
        feature.astype('<f4').tofile(os.path.join(path, basename+'.dat'))
    return True


class ImageDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.img_paths = glob.glob(os.path.join(dir, '*.*'))
        assert(len(self.img_paths) != 0)
        self.transform = T.Compose([T.Resize([256, 128]), T.ToTensor()]) if transform is None else transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.img_paths[index]))
        basename = get_file_basename(self.img_paths[index])
        return img, basename

@torch.no_grad()
def extract(root=''):
    img_dir = os.path.join(root, 'image')
    fea_dir = os.path.join(root, 'feature')
    os.makedirs(fea_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(img_dir, transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=8)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ae_net = AEModel('efficientnet_b4(num_classes={})', extractor_out_dim=DIM_NUM, compress_dim=32)
    ae_net.load_param(os.path.join(root, f'project/Net_best.pth'))
    ae_net.to(device)
    ae_net.eval()
    for imgs, basenames in dataloader:
        imgs = imgs.to(device)
        features = ae_net.extract(imgs)
        features = features.cpu()
        zfeatrues = torch.zeros(features.shape[0], 2048, dtype=features.dtype)
        zfeatrues[:, :DIM_NUM] = features
        write_feature_file(zfeatrues.numpy(), basenames, fea_dir)

    print('Extraction Done')
