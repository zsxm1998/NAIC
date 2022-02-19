import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Conv1d(1, 64, 7, 2)
        self.en1 = nn.BatchNorm1d(64)
        self.el2 = nn.Conv1d(64, 128, 5, 2)
        self.en2 = nn.BatchNorm1d(128)
        self.el3 = nn.Conv1d(128, 256, 3, 2)
        self.en3 = nn.BatchNorm1d(256)
        self.el5 = nn.Linear(14336, intermediate_dim)

    def forward(self, x):
        x = self.relu(self.en1(self.el1(x)))
        x = self.relu(self.en2(self.el2(x)))
        x = self.relu(self.en3(self.el3(x)))
        x = torch.flatten(x, 1)
        return self.el5(x)


class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU(inplace=True)
        self.dl1 = nn.Linear(intermediate_dim, 256)
        self.dn1 = nn.LayerNorm(256)
        self.dl2 = nn.Linear(256, 512)
        self.dn2 = nn.LayerNorm(512)
        self.dl3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        out = self.dl3(x)
        return out


class AutoEncoderConv(nn.Module):
    def __init__(self, intermediate_dim, input_dim, output_dim):
        super(AutoEncoderConv, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = Encoder(intermediate_dim, input_dim)
        self.decoder = Decoder(intermediate_dim, output_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))