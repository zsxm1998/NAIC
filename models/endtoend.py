import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random

from .backbones.efficientnet import efficientnet_b4
from .backbones.resnet import resnet50
from .backbones.resnet_nl import resnet50_nl


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


class EndtoEndModel(nn.Module):
    def __init__(self, model_name, id_num, extractor_out_dim, compress_dim):
        super(EndtoEndModel, self).__init__()
        #self.extractor = eval(model_name)(num_classes=extractor_out_dim, pretrained='.details/checkpoints/efficientnet_b4.pth')
        self.extractor = eval(model_name.format(extractor_out_dim))
        self.encoder = Encoder(compress_dim, extractor_out_dim)
        self.decoder = Decoder(compress_dim, extractor_out_dim)
        #self.BNNeck = nn.Sequential(nn.BatchNorm1d(extractor_out_dim, affine=False), nn.Linear(extractor_out_dim, id_num, bias=False))
        self.BNNeck = nn.Sequential(nn.BatchNorm1d(extractor_out_dim), nn.Linear(extractor_out_dim, id_num, bias=False))
        self.BNNeck[0].bias.requires_grad_(False)

    def forward(self, imgs):
        ft = self.extractor(imgs)
        fi = self.BNNeck(ft)
        fr = self.decoder(self.encoder(ft).half().float()) if random() < 0.5 else self.decoder(self.encoder(ft))
        return ft, fi, fr