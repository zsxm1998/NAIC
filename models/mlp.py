import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Encoder(nn.Module):
    def __init__(self, input_dim=2048):
        super(MLP_Encoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, 512)
        self.en1 = nn.LayerNorm(512)
        self.el2 = nn.Linear(512, 256)
        self.en2 = nn.LayerNorm(256)
        self.el3 = nn.Linear(256, 128)
        self.en3 = nn.LayerNorm(128)

    def forward(self, x):
        x = self.relu(self.en1(self.el1(x)))
        x = self.relu(self.en2(self.el2(x)))
        out = self.relu(self.en3(self.el3(x)))
        return out


class MLP_Decoder(nn.Module):
    def __init__(self, output_dim=2048):
        super(MLP_Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dl1 = nn.Linear(128, 256)
        self.dn1 = nn.LayerNorm(256)
        self.dl2 = nn.Linear(256, 512)
        self.dn2 = nn.LayerNorm(512)
        self.dl3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        out = self.dl3(x)
        return out


class MLP_AutoEncoder(nn.Module):
    def __init__(self, vector_dim=2048):
        super(MLP_AutoEncoder, self).__init__()
        self.encoder = MLP_Encoder(vector_dim)
        self.decoder = MLP_Decoder(vector_dim)

    def forward(self, x):
        m = self.encoder(x)
        out = self.decoder(m)
        return out, m


class MLP_MoCo(nn.Module):
    def __init__(self, vector_dim, base_encoder=MLP_AutoEncoder, dim=128, K=65536, m=0.999):
        super(MLP_MoCo, self).__init__()
        self.K = K
        self.m = m
        self.encoder_q = base_encoder(vector_dim)
        self.encoder_k = base_encoder(vector_dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_label", torch.ones(K)*-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_label):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr+batch_size <= self.K:
            self.queue[ptr:ptr+batch_size] = keys
            self.queue_label[ptr:ptr+batch_size] = keys_label
        else:
            diff = self.K - ptr
            self.queue[ptr:] = keys[:diff]
            self.queue_label[ptr:] = keys_label[:diff]
            self.queue[:batch_size-diff] = keys[diff:]
            self.queue_label[:batch_size-diff] = keys_label[diff:]
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, input_q, input_k, k_label):
        reconstruct_q, q = self.encoder_q(input_q)
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            reconstruct_k, k = self.encoder_k(input_k)
            k = F.normalize(k, dim=1)
        queue = self.queue.clone().detach()
        queue_label = self.queue_label.clone().detach()
        self._dequeue_and_enqueue(k, k_label)
        return q, k, queue, queue_label, reconstruct_q, reconstruct_k
