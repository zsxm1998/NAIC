import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, concat=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.concat = concat

    def forward(self, q, q_label, k, k_label):
        q_label = q_label.contiguous().view(-1,1)
        k_label = k_label.contiguous().view(-1,1)
        if self.concat:
            k = torch.cat([q, k], dim=0)
            k_label = torch.cat([q_label, k_label], dim=0)
        
        mask = torch.eq(q_label, k_label.T).float().to(q.device)
        logits = torch.matmul(q, k.T) / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits)
        if self.concat:
            mask = torch.scatter(mask, 1, torch.arange(mask.shape[0]).view(-1,1).to(mask.device), 0)
            exp_logits = torch.scatter(exp_logits, 1, torch.arange(exp_logits.shape[0]).view(-1,1).to(exp_logits.device), 0)

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()
        return loss