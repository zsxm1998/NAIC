import torch.nn as nn
import torch.nn.functional as F


# class Encoder(nn.Module):
#     def __init__(self, intermediate_dim, input_dim):
#         super(Encoder, self).__init__()
#         self.intermediate_dim = intermediate_dim
#         self.input_dim = input_dim

#         self.relu = nn.ReLU(inplace=True)
#         self.el1 = nn.Linear(input_dim, 4096)
#         self.en1 = nn.BatchNorm1d(4096)
#         self.dp1 = nn.Dropout(p=0.5)
#         self.el2 = nn.Linear(4096, 2048)
#         self.en2 = nn.BatchNorm1d(2048)
#         self.dp2 = nn.Dropout(p=0.3)
#         self.el3 = nn.Linear(2048, 1024)
#         self.en3 = nn.BatchNorm1d(1024)
#         self.el4 = nn.Linear(1024, 512)
#         self.en4 = nn.BatchNorm1d(512)
#         self.el5 = nn.Linear(512, intermediate_dim)
#         self.en5 = nn.BatchNorm1d(intermediate_dim)

#     def forward(self, x):
#         x = self.dp1(self.relu(self.en1(self.el1(x))))
#         x = self.dp2(self.relu(self.en2(self.el2(x))))
#         x = self.relu(self.en3(self.el3(x)))
#         x = self.relu(self.en4(self.el4(x)))
#         out = self.en5(self.el5(x))
#         return out

# class Decoder(nn.Module):
#     def __init__(self, intermediate_dim, output_dim=463):
#         super(Decoder, self).__init__()
#         self.intermediate_dim = intermediate_dim
#         self.output_dim = output_dim

#         self.relu = nn.ReLU(inplace=True)
#         self.dl1 = nn.Linear(intermediate_dim, 512)
#         self.dn1 = nn.BatchNorm1d(512)
#         self.dl2 = nn.Linear(512, 1024)
#         self.dn2 = nn.BatchNorm1d(1024)
#         self.dl3 = nn.Linear(1024, 2048)
#         self.dn3 = nn.BatchNorm1d(2048)
#         self.dl4 = nn.Linear(2048, 4096)
#         self.dn4 = nn.BatchNorm1d(4096)
#         self.dl5 = nn.Linear(4096, output_dim)

#     def forward(self, x):
#         x = self.relu(self.dn1(self.dl1(x)))
#         x = self.relu(self.dn2(self.dl2(x)))
#         x = self.relu(self.dn3(self.dl3(x)))
#         x = self.relu(self.dn4(self.dl4(x)))
#         out = self.dl5(x)
#         return out


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, 8192)
        self.en1 = nn.BatchNorm1d(8192)
        self.dp1 = nn.Dropout(p=0.5)
        self.el2 = nn.Linear(8192, 4096)
        self.en2 = nn.BatchNorm1d(4096)
        self.dp2 = nn.Dropout(p=0.3)
        self.el3 = nn.Linear(4096, 2048)
        self.en3 = nn.BatchNorm1d(2048)
        self.el4 = nn.Linear(2048, 1024)
        self.en4 = nn.BatchNorm1d(1024)
        self.el5 = nn.Linear(1024, 512)
        self.en5 = nn.BatchNorm1d(512)
        self.el6 = nn.Linear(512, intermediate_dim)
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
        self.dl1 = nn.Linear(intermediate_dim, 512)
        self.dn1 = nn.BatchNorm1d(512)
        self.dl2 = nn.Linear(512, 1024)
        self.dn2 = nn.BatchNorm1d(1024)
        self.dl3 = nn.Linear(1024, 2048)
        self.dn3 = nn.BatchNorm1d(2048)
        self.dl4 = nn.Linear(2048, 4096)
        self.dn4 = nn.BatchNorm1d(4096)
        self.dl5 = nn.Linear(4096, 8192)
        self.dn5 = nn.BatchNorm1d(8192)
        self.dl6 = nn.Linear(8192, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        x = self.relu(self.dn3(self.dl3(x)))
        x = self.relu(self.dn4(self.dl4(x)))
        x = self.relu(self.dn5(self.dl5(x)))
        out = self.dl6(x)
        return out


class AutoEncoderMLP(nn.Module):
    def __init__(self, intermediate_dim, input_dim, output_dim):
        super(AutoEncoderMLP, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = Encoder(intermediate_dim, input_dim)
        self.decoder = Decoder(intermediate_dim, output_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x).half().float())