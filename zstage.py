import torch
import torch.nn as nn
from models.autoencoder import Decoder

class Decoder1(nn.Module):
    def __init__(self, intermediate_dim, output_dim):
        super(Decoder1, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU(inplace=True)
        self.dl1 = nn.Linear(intermediate_dim, 512)
        self.dn1 = nn.BatchNorm1d(512)
        self.dl2 = nn.Linear(512, 1024)
        self.dn2 = nn.BatchNorm1d(1024)
        self.dl3 = nn.Linear(1024, 2048)
        self.dn3 = nn.BatchNorm1d(2048)
        self.dl4 = nn.Linear(2048, 4096)
        self.dn4 = nn.BatchNorm1d(4096)
        self.dl5 = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        x = self.relu(self.dn3(self.dl3(x)))
        x = self.relu(self.dn4(self.dl4(x)))
        out = self.dl5(x)
        return out

net_origin = Decoder1(32, 463)
net_origin.load_state_dict(torch.load('/nfs3-p2/zsxm/zstage/Decoder_128.pth'))

net_save = Decoder(32, 463)
net_save.dl0.load_state_dict(net_origin.dl1.state_dict())
net_save.dn0.load_state_dict(net_origin.dn1.state_dict())
net_save.dl1.load_state_dict(net_origin.dl2.state_dict())
net_save.dn1.load_state_dict(net_origin.dn2.state_dict())
net_save.dl2.load_state_dict(net_origin.dl3.state_dict())
net_save.dn2.load_state_dict(net_origin.dn3.state_dict())
net_save.dl3.load_state_dict(net_origin.dl4.state_dict())
net_save.dn3.load_state_dict(net_origin.dn4.state_dict())
net_save.dl4.load_state_dict(net_origin.dl5.state_dict())

# torch.save(net_save.state_dict(), '/nfs3-p2/zsxm/zstage/Decoder.pth')
print(net_save)