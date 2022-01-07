import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_AutoEncoder(nn.Module):
    def __init__(self):
        super(MLP_AutoEncoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(2048, 1024)
        self.en1 = nn.LayerNorm(1024)
        self.el2 = nn.Linear(1024, 512)
        self.en2 = nn.LayerNorm(512)
        self.el3 = nn.Linear(512, 256)
        self.en3 = nn.LayerNorm(256)

        self.dl1 = nn.Linear(256, 512)
        self.dn1 = nn.LayerNorm(512)
        self.dl2 = nn.Linear(512, 1024)
        self.dn2 = nn.LayerNorm(1024)
        self.dl3 = nn.Linear(1024, 2048)

    def forward(self, x):
        x = self.relu(self.en1(self.el1(x)))
        x = self.relu(self.en2(self.el2(x)))
        m = self.relu(self.en3(self.el3(x)))

        out = self.relu(self.dn1(self.dl1(m)))
        out = self.relu(self.dn2(self.dl2(out)))
        out = self.dl3(out)
        return out, m
