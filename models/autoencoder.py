import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, 16384)
        self.en1 = nn.BatchNorm1d(16384)
        self.dp1 = nn.Dropout(p=0.5)
        self.el2 = nn.Linear(16384, 8192)
        self.en2 = nn.BatchNorm1d(8192)
        self.dp2 = nn.Dropout(p=0.3)
        self.el3 = nn.Linear(8192, 4096)
        self.en3 = nn.BatchNorm1d(4096)
        self.el4 = nn.Linear(4096, 2048)
        self.en4 = nn.BatchNorm1d(2048)
        self.el5 = nn.Linear(2048, 1024)
        self.en5 = nn.BatchNorm1d(1024)
        self.el6 = nn.Linear(1024, intermediate_dim)
        self.en6 = nn.BatchNorm1d(intermediate_dim)

    def forward(self, x):
        x = self.dp1(self.relu(self.en1(self.el1(x))))
        x = self.dp2(self.relu(self.en2(self.el2(x))))
        x = self.relu(self.en3(self.el3(x)))
        x = self.relu(self.en4(self.el4(x)))
        x = self.relu(self.en5(self.el5(x)))
        out = self.en6(self.el6(x))
        return out

class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU(inplace=True)
        self.dl1 = nn.Linear(intermediate_dim, 1024)
        self.dn1 = nn.BatchNorm1d(1024)
        self.dl2 = nn.Linear(1024, 2048)
        self.dn2 = nn.BatchNorm1d(2048)
        self.dl3 = nn.Linear(2048, 4096)
        self.dn3 = nn.BatchNorm1d(4096)
        self.dl4 = nn.Linear(4096, 8192)
        self.dn4 = nn.BatchNorm1d(8192)
        self.dl5 = nn.Linear(8192, 16384)
        self.dn5 = nn.BatchNorm1d(16384)
        self.dl6 = nn.Linear(16384, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        x = self.relu(self.dn3(self.dl3(x)))
        x = self.relu(self.dn4(self.dl4(x)))
        x = self.relu(self.dn5(self.dl5(x)))
        out = self.dl6(x)
        return out


# class Encoder(nn.Module):
#     def __init__(self, intermediate_dim, input_dim):
#         super(Encoder, self).__init__()
#         self.intermediate_dim = intermediate_dim
#         self.input_dim = input_dim

#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool1d(2, 2)
#         self.el1 = nn.Linear(input_dim, 512)
#         self.en1 = nn.LayerNorm(512)
#         self.dp1 = nn.Dropout(p=0.1)
#         self.el2 = nn.Linear(512, 256)
#         self.en2 = nn.LayerNorm(256)
#         self.el3 = nn.Linear(256, 128)
#         self.en3 = nn.LayerNorm(128)
#         self.el4 = nn.Linear(128, intermediate_dim)
#         self.en4 = nn.LayerNorm(intermediate_dim)

#     def forward(self, x):
#         x = self.dp1(self.relu(self.en1(self.el1(x))))
#         y2 = self.en2(self.el2(x))
#         x = self.relu(self.avgpool(x) + y2)
#         y3 = self.en3(self.el3(x))
#         x = self.relu(self.avgpool(x) + y3)
#         out = self.en4(self.el4(x))
#         return out

# class Decoder(nn.Module):
#     def __init__(self, intermediate_dim, output_dim):
#         super(Decoder, self).__init__()
#         self.intermediate_dim = intermediate_dim
#         self.output_dim = output_dim

#         self.relu = nn.ReLU(inplace=True)
#         self.dl1 = nn.Linear(intermediate_dim, 128)
#         self.dn1 = nn.LayerNorm(128)
#         self.dl2 = nn.Linear(128, 256)
#         self.dn2 = nn.LayerNorm(256)
#         self.dl3 = nn.Linear(256, 512)
#         self.dn3 = nn.LayerNorm(512)
#         self.dl4 = nn.Linear(512, output_dim)

#     def forward(self, x):
#         x = self.relu(self.dn1(self.dl1(x)))
#         y2 = self.dn2(self.dl2(x))
#         x = self.relu(F.interpolate(x.unsqueeze(1), scale_factor=2).squeeze(1) + y2)
#         y3 = self.dn3(self.dl3(x))
#         x = self.relu(F.interpolate(x.unsqueeze(1), scale_factor=2).squeeze(1) + y3)
#         out = self.dl4(x)
#         return out


# class Encoder(nn.Module):
#     def __init__(self, intermediate_dim, input_dim):
#         super(Encoder, self).__init__()
#         self.intermediate_dim = intermediate_dim
#         self.input_dim = input_dim

#         self.relu = nn.ReLU(inplace=True)
#         self.el1 = nn.Linear(input_dim, 512)
#         self.en1 = nn.LayerNorm(512)
#         self.dp1 = nn.Dropout(p=0.3)
#         self.el2 = nn.Linear(512, 256)
#         self.en2 = nn.LayerNorm(256)
#         self.dp2 = nn.Dropout(p=0.1)
#         self.el3 = nn.Linear(256, 128)
#         self.en3 = nn.LayerNorm(128)
#         self.el4 = nn.Linear(128, 64)
#         self.en4 = nn.LayerNorm(64)
#         self.el5 = nn.Linear(64, intermediate_dim)
#         self.en5 = nn.LayerNorm(intermediate_dim)

#     def forward(self, x):
#         x = self.dp1(self.relu(self.en1(self.el1(x))))
#         x = self.dp2(self.relu(self.en2(self.el2(x))))
#         x = self.relu(self.en3(self.el3(x)))
#         x = self.relu(self.en4(self.el4(x)))
#         out = self.en5(self.el5(x))
#         return out

# class Decoder(nn.Module):
#     def __init__(self, intermediate_dim, output_dim):
#         super(Decoder, self).__init__()
#         self.intermediate_dim = intermediate_dim
#         self.output_dim = output_dim

#         self.relu = nn.ReLU(inplace=True)
#         self.dl = nn.Linear(intermediate_dim, 64)
#         self.dn = nn.LayerNorm(64)
#         self.dl0 = nn.Linear(64, 128)
#         self.dn0 = nn.LayerNorm(128)
#         self.dl1 = nn.Linear(128, 256)
#         self.dn1 = nn.LayerNorm(256)
#         self.dl2 = nn.Linear(256, 512)
#         self.dn2 = nn.LayerNorm(512)
#         self.dl3 = nn.Linear(512, output_dim)

#     def forward(self, x):
#         x = self.relu(self.dn(self.dl(x)))
#         x = self.relu(self.dn0(self.dl0(x)))
#         x = self.relu(self.dn1(self.dl1(x)))
#         x = self.relu(self.dn2(self.dl2(x)))
#         out = self.dl3(x)
#         return out


class AutoEncoderMLP(nn.Module):
    def __init__(self, intermediate_dim, input_dim, output_dim):
        super(AutoEncoderMLP, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = Encoder(intermediate_dim, input_dim)
        self.decoder = Decoder(intermediate_dim, output_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))