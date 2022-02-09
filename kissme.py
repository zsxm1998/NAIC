import os
import torch
from tqdm import tqdm

NAME_NUM = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
BATCH_SIZE = 5000
START = 16756
END = 34759

save_dir = '/nfs3-p1/zsxm/naic/preliminary/train/'

train_reshape = torch.load('/nfs3-p1/zsxm/naic/preliminary/train/train_reshape.pt')
train_label = torch.load('/nfs3-p1/zsxm/naic/preliminary/train/train_label.pt')
print(train_reshape.shape, train_label.shape)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_reshape = train_reshape.to(device)
train_label = train_label.to(device)

sigma0 = torch.zeros(train_reshape.shape[1], train_reshape.shape[1], dtype=torch.float, device=device)
sigma1 = torch.zeros(train_reshape.shape[1], train_reshape.shape[1], dtype=torch.float, device=device)
for i in tqdm(range(START, END)):
    xij = train_reshape[i] - train_reshape[i+1:]
    flag_0 = train_label[i+1:].ne(train_label[i])
    flag_1 = train_label[i+1:].eq(train_label[i])
    for k in range(0, xij.shape[0], BATCH_SIZE):
        mij = torch.bmm(xij[k:k+BATCH_SIZE].unsqueeze(-1), xij[k:k+BATCH_SIZE].unsqueeze(1))
        sigma0 += (flag_0[k:k+BATCH_SIZE, None, None]*mij).sum(dim=0)
        sigma1 += (flag_1[k:k+BATCH_SIZE, None, None]*mij).sum(dim=0)
        del mij

sigma0 *= 2
sigma1 *= 2
torch.save(sigma0, os.path.join(save_dir, f'sigma0_{NAME_NUM}.pt'))
torch.save(sigma1, os.path.join(save_dir, f'sigma1_{NAME_NUM}.pt'))