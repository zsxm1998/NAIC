import os
import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F

from .efficientnet import efficientnet_b4, efficientnet_b5


DIM_NUM = 128
BATCH_SIZE = 512


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

    # val_dataset = ImageDataset(img_dir)
    # val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=8)
    # img_count = 0
    # rgb = torch.zeros(3)
    # for imgs, basenames in val_dataloader:
    #     img_count += imgs.shape[0]
    #     rgb += imgs.sum(dim=(0,2,3))
    # img_count *= 128*256
    # rgb /= img_count
    # mean = rgb
    # rgb = rgb[None, :, None, None]
    # var = torch.zeros(3)
    # for imgs, basenames in val_dataloader:
    #     var += torch.pow(imgs - rgb, 2).sum(dim=(0,2,3))
    # var /= img_count
    # std = var.sqrt()
    # mean = mean.tolist()
    # std = std.tolist()

    transform = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(img_dir, transform=transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=8)
    extractor = efficientnet_b5(num_classes=DIM_NUM)
    extractor.load_state_dict(torch.load(os.path.join(root, f'project/Extractor_{DIM_NUM}_best.pth')))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    extractor.to(device)
    extractor.eval()
    for imgs, basenames in dataloader:
        imgs = imgs.to(device)
        features = extractor(imgs)
        features = features.cpu()
        zfeatrues = torch.zeros(features.shape[0], 2048, dtype=features.dtype)
        zfeatrues[:, :DIM_NUM] = features
        write_feature_file(zfeatrues.numpy(), basenames, fea_dir)
        # features = features.cpu().numpy()
        # write_feature_file(features, basenames, fea_dir)

    print('Extraction Done')
