import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random

from .backbones.efficientnet import efficientnet_b4, efficientnet_b5
from .backbones.resnet import resnet50
from .backbones.resnet_nl import resnet50_nl


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ExtractorModel(nn.Module):
    def __init__(self, model_name, id_num, extractor_out_dim):
        super(ExtractorModel, self).__init__()
        self.extractor = eval(model_name.format(extractor_out_dim))
        # self.BNNeck = nn.Sequential(nn.BatchNorm1d(extractor_out_dim), nn.Linear(extractor_out_dim, id_num, bias=False))
        # self.BNNeck[0].bias.requires_grad_(False)
        self.bottleneck = nn.BatchNorm1d(extractor_out_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.train_classifier = nn.Linear(extractor_out_dim, id_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.train_classifier.apply(weights_init_classifier)

    def forward(self, imgs):
        ft = self.extractor(imgs)
        fb = self.bottleneck(ft)
        fi = self.train_classifier(fb)
        return ft, fb, fi

    def extract(self, imgs, after=False):
        ft = self.extractor(imgs)
        if after:
            return self.bottleneck(ft)
        return ft

    def load_param(self, state_path):
        param_dict = torch.load(state_path)
        for key in param_dict:
            if 'train_classifier' in key:
                continue
            self.state_dict()[key].copy_(param_dict[key])