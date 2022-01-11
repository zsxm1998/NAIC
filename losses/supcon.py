import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, q_label, k, k_label):
        q_label = q_label.contiguous().view(-1,1)
        k_label = k_label.contiguous().view(-1,1)
        mask = torch.eq(q_label, k_label.T).float().to(q.device)

        logits = torch.matmul(q, k.T) / self.temperature
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()
        return loss