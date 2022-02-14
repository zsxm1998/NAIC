import os
import numpy as np
import torch
import tables
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

work_dir = '/nfs3-p2/zsxm/naic/preliminary/test_B/k_reciprocal/'
ALL_NUM = 220939
PROBE_NUM = 10000
BS = 200
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
RSTART = 0
REND = 5


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LoadV(Dataset):
    def __init__(self, dir):
        self.v_hdf5_file = tables.open_file(dir, mode='r')
        self.V = self.v_hdf5_file.root.V

    def __len__(self):
        return self.V.shape[0]-PROBE_NUM

    def __getitem__(self, index):
        return torch.from_numpy(self.V[index+PROBE_NUM])

    def __del__(self):
        self.v_hdf5_file.close()

def calc_jaccard(num, start, end):
    v = LoadV(os.path.join(work_dir, f'V.hdf5'))
    v_loader = DataLoader(v, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)
    #jdist_hdf5_file = tables.open_file(os.path.join(work_dir, f'jaccard_dist_{num}.hdf5'), mode='w')
    #jaccard_dist = jdist_hdf5_file.create_earray(jdist_hdf5_file.root, 'jaccard_dist', tables.Float32Atom(), shape=(end-start, 0), filters=tables.Filters(), expectedrows=ALL_NUM-PROBE_NUM)
    jaccard_dist = []

    probe = torch.from_numpy(v.V[start: end]).unsqueeze_(1).to(device)
    for gallery in tqdm(v_loader, desc=f'calculate jaccard distance of {start}-{end}'):
        gallery = gallery.to(device)
        jaccard = torch.min(probe, gallery).sum(dim=-1) / torch.max(probe, gallery).sum(dim=-1)
        jaccard_dist.append(jaccard.cpu()) #jaccard_dist.append(jaccard.cpu().numpy())
    
    #jdist_hdf5_file.close()
    jaccard_dist = torch.cat(jaccard_dist, dim=1)
    torch.save(jaccard_dist, os.path.join(work_dir, f'jaccard_dist_{num}.pt'))

class LoadDist(Dataset):
    def __init__(self, jaccard_dir, origin_dir, num, bs, lmd=0.3):
        self.jaccard_dist = torch.load(jaccard_dir)
        self.origin_hdf5_file = tables.open_file(origin_dir, mode='r')
        self.original_dist = self.origin_hdf5_file.root.original_dist
        self.start = num * bs
        self.lmd = lmd

    def __len__(self):
        return self.jaccard_dist.shape[0]

    def __getitem__(self, index):
        return index+self.start, self.lmd * torch.from_numpy(self.original_dist[index+self.start, PROBE_NUM:]) + (1-self.lmd) * self.jaccard_dist[index]

    def __del__(self):
        self.origin_hdf5_file.close()

query_feature_B_dir = '/nfs3-p2/zsxm/naic/preliminary/test_B/query_feature_B'
gallery_feature_B_dir = '/nfs3-p2/zsxm/naic/preliminary/test_B/gallery_feature_B'
query_names = sorted(os.listdir(query_feature_B_dir))
gallery_names = sorted(os.listdir(gallery_feature_B_dir))

def calc_result(num, start, end):
    batch_size = 16
    jaccard_dir = os.path.join(work_dir, f'jaccard_dist_{num}.pt')
    origin_dir = os.path.join(work_dir, 'original_dist_pearson.hdf5')
    fdataset = LoadDist(jaccard_dir, origin_dir, num, BS)
    floader = DataLoader(fdataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    res_dict = {}
    for index, fdist in tqdm(floader, desc=f'calc result {num}-{start}-{end}'):
        idx = torch.argsort(fdist, dim=-1, descending=True)
        for i in range(len(idx)):
            name = query_names[index[i]]
            query_res = []
            for j in range(100):
                query_res.append(gallery_names[idx[i, j]])
            res_dict[name] = query_res
    
    torch.save(res_dict, os.path.join(work_dir, f'res_{num}.pt'))

for i in range(RSTART, REND):
    if i != RSTART:
        calc_jaccard(i, i*BS, (i+1)*BS)
    calc_result(i, i*BS, (i+1)*BS)