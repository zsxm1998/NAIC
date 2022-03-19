import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

from .efficientnet import efficientnet_b4, efficientnet_b5


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


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.prelu = nn.PReLU()
        self.linear = nn.Linear(input_dim, intermediate_dim)

    def forward(self, x):
        return self.prelu(self.linear(x))


class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(intermediate_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class AEModel(nn.Module):
    byte_rate_category = 3
    byte_rate_base = 64

    def __init__(self, model_name, extractor_out_dim, compress_dim, id_num=1000, neck_feat=True):
        super(AEModel, self).__init__()
        self.neck_feat = neck_feat
        self.extractor = eval(model_name.format(extractor_out_dim))
        self.encoders = nn.ModuleList([Encoder(compress_dim * 2**i, extractor_out_dim) for i in range(self.byte_rate_category)])
        self.decoders = nn.ModuleList([Decoder(compress_dim * 2**i, extractor_out_dim) for i in range(self.byte_rate_category)])
        self.bottlenecks = nn.ModuleList([nn.BatchNorm1d(extractor_out_dim) for _ in range(self.byte_rate_category)])
        for bottleneck in self.bottlenecks:
            bottleneck.bias.requires_grad_(False)
        self.train_classifiers = nn.ModuleList([nn.Linear(extractor_out_dim, id_num, bias=False) for _ in range(self.byte_rate_category)])
        self.bottlenecks.apply(weights_init_kaiming)
        self.train_classifiers.apply(weights_init_classifier)

    def forward(self, imgs):
        f_e = self.extractor(imgs)
        f_g = [decoder(encoder(f_e).half().float()) for encoder, decoder in zip(self.encoders, self.decoders)]
        f_f = [bottleneck(f_g[i]) for i, bottleneck in enumerate(self.bottlenecks)]
        if self.training:
            f_c = [classifier(f_f[i]) for i, classifier in enumerate(self.train_classifiers)]
            return f_g, f_c
        else:
            if self.neck_feat:
                return f_f
            else:
                return f_g

    def extract(self, imgs):
        return self.extractor(imgs)

    def compress(self, feas, byte_rate):
        i = int(log2(byte_rate//self.byte_rate_base))
        return self.encoders[i](feas)

    def reconstract(self, feas, byte_rate):
        i = int(log2(byte_rate//self.byte_rate_base))
        return self.decoders[i](feas)

    def reid(self, feas, byte_rate):
        i = int(log2(byte_rate//self.byte_rate_base))
        return self.bottlenecks[i](feas)

    def load_param(self, state_path):
        param_dict = torch.load(state_path)
        for key in param_dict:
            if 'train_classifiers' in key:
                continue
            self.state_dict()[key].copy_(param_dict[key])
