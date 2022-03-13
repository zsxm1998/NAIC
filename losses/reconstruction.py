import torch
import torch.nn as nn
import torch.nn.functional as F


class L2ReconstructionLoss(nn.Module):
    def __init__(self):
        super(L2ReconstructionLoss, self).__init__()

    def forward(self, reconstructed, origin):
        assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
        return torch.linalg.norm(reconstructed-origin, dim=1, ord=2).mean()


class ExpReconstructionLoss(nn.Module):
    def __init__(self):
        super(ExpReconstructionLoss, self).__init__()

    def forward(self, reconstructed, origin):
        assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
        return reconstructed.sub(origin).pow(2).exp().mean()


class EnlargeReconstructionLoss(nn.Module):
    def __init__(self):
        super(EnlargeReconstructionLoss, self).__init__()
        self.l2loss = nn.MSELoss()
        self.EXP = 1/3

    def forward(self, reconstructed, origin):
        assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
        reconstructed = reconstructed.sign()*reconstructed.abs().pow(self.EXP)
        origin = origin.sign()*origin.abs().pow(self.EXP)
        return self.l2loss(reconstructed, origin)


class NormalizeReconstructionLoss(nn.Module):
    def __init__(self):
        super(NormalizeReconstructionLoss, self).__init__()
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, reconstructed, origin):
        assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
        mse = self.mseloss(reconstructed, origin)
        loss = (mse / (origin**2+1e-15)).mean()
        return loss


class CosineReconstructionLoss(nn.Module):
    def __init__(self):
        super(CosineReconstructionLoss, self).__init__()
        self.mseloss = nn.MSELoss(reduction='none')

    def cosine(self, input, target):
        input = F.normalize(input, dim=-1)
        target = F.normalize(target, dim=-1)
        loss = torch.einsum('bi,bi->b', input, target)
        loss = (-loss+1) / 2 + 0.01
        return loss

    def forward(self, reconstructed, origin):
        assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
        mse = self.mseloss(reconstructed, origin)
        mse = (mse / (origin**2+1e-15)).sum(dim=-1)
        cos = self.cosine(reconstructed, origin)
        loss = mse * cos
        return loss.mean()
