# from zstage.project.extract import resnet_encoder
# from zstage.project.compress import Encoder
# from zstage.project.reconstruct import Decoder
# import torch

# extractor = resnet_encoder(50, n_outputs=128)
# encoder = Encoder(32, 128)
# decoder = Decoder(32, 128)

# extractor.load_state_dict(torch.load('zstage/project/Extractor_128_best.pth'))
# encoder.load_state_dict(torch.load('zstage/project/Encoder_32_best.pth'))
# decoder.load_state_dict(torch.load('zstage/project/Decoder_32_best.pth'))

import torch
from models.ae import AEModel

m = AEModel("efficientnet_b4(num_classes={}, pretrained='.details/checkpoints/efficientnet_b4.pth')", 1024, 32)
for k in m.state_dict().keys():
    if 'num' in k:
        print(k)