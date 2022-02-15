import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Conv1d(1, 64, 11, 2)
        self.en1 = nn.BatchNorm1d(64)
        self.el2 = nn.Conv1d(64, 128, 13, 2)
        self.en2 = nn.BatchNorm1d(128)
        self.el3 = nn.Conv1d(128, 256, 15, 2)
        self.en3 = nn.BatchNorm1d(256)
        self.el4 = nn.Conv1d(256, 512, 17, 2)
        self.en4 = nn.BatchNorm1d(512)
        self.el5 = nn.LazyLinear(intermediate_dim, bias=False)

    def forward(self, x):
        x = self.relu(self.en1(self.el1(x)))
        x = self.relu(self.en2(self.el2(x)))
        x = self.relu(self.en3(self.el3(x)))
        x = self.relu(self.en4(self.el4(x)))
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


class AutoEncoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = Encoder(intermediate_dim, input_dim)
        self.decoder = Decoder(intermediate_dim, output_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))