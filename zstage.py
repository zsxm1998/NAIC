import torch
import torch.nn as nn
from models.embedding import MoCo, resnet_encoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
from torchvision import transforms as T
from tqdm import tqdm

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def write_feature_file(features, basenames, path):
    assert len(features) == len(basenames)
    for feature, basename in zip(features, basenames):
        feature.astype('<f4').tofile(os.path.join(path, basename+'.dat'))
    return True

class ImageDataset(Dataset):
    def __init__(self, dir):
        self.img_paths = glob.glob(os.path.join(dir, '*.*'))
        assert(len(self.img_paths) != 0)
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.img_paths[index]))
        basename = get_file_basename(self.img_paths[index])
        return img, basename

dataset = ImageDataset('/nfs3-p2/zsxm/naic/rematch/train/train_picture')
dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=8)

fea_dir = '/nfs3-p2/zsxm/naic/rematch/train/features'
os.makedirs(fea_dir, exist_ok=True)

encoder = resnet_encoder(34)
encoder.load_state_dict(torch.load('zstage/project/Encoder_best.pth'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
encoder.eval()
fl = []
with torch.no_grad():
    for imgs, basenames in tqdm(dataloader):
        imgs = imgs.to(device)
        features = encoder(imgs)
        fl.append(features.cpu())

fl = torch.cat(fl)
print(fl.shape)
torch.save(fl, '/nfs3-p2/zsxm/naic/rematch/train/features.pth')