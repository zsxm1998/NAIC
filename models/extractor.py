import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random

from .backbones.efficientnet import efficientnet_b4, efficientnet_b5
from .backbones.resnet import resnet50
from .backbones.resnet_nl import resnet50_nl


class ExtractorModel(nn.Module):
    def __init__(self, model_name, id_num, extractor_out_dim):
        super(ExtractorModel, self).__init__()
        self.extractor = eval(model_name.format(extractor_out_dim))
        self.BNNeck = nn.Sequential(nn.BatchNorm1d(extractor_out_dim), nn.Linear(extractor_out_dim, id_num, bias=False))
        self.BNNeck[0].bias.requires_grad_(False)

    def forward(self, imgs):
        ft = self.extractor(imgs)
        fi = self.BNNeck(ft)
        return ft, fi