import torch
import torch.nn as nn
import torch.nn.functional as F


class L2ReconstructionLoss(nn.Module):
    def __init__(self):
        super(L2ReconstructionLoss, self).__init__()

    def forward(self, reconstructed, origin):
        assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
        return torch.linalg.vector_norm(reconstructed-origin, dim=1, ord=2).mean()


class ExpReconstructionLoss(nn.Module):
    def __init__(self):
        super(ExpReconstructionLoss, self).__init__()

    def forward(self, reconstructed, origin):
        return reconstructed.sub(origin).pow(2).exp().mean()


class EnlargeReconstructionLoss(nn.Module):
    def __init__(self):
        super(EnlargeReconstructionLoss, self).__init__()
        self.l2loss = nn.MSELoss()
        self.EXP = 1/3

    def forward(self, reconstructed, origin):
        reconstructed = reconstructed.sign()*reconstructed.abs().pow(self.EXP)
        origin = origin.sign()*origin.abs().pow(self.EXP)
        return self.l2loss(reconstructed, origin)
