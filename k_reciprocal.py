import os
import numpy as np
import torch
import tables
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json

ALL_NUM = 448794
PROBE_NUM = 20000
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
work_dir = '/nfs3-p2/zsxm/naic/preliminary/test_A/k_reciprocal/'

def calc_V(num, start, end, k1=100):
    dist_hdf5_file = tables.open_file(os.path.join(work_dir, 'original_dist_pearson.hdf5'), mode='r')
    original_dist = dist_hdf5_file.root.original_dist
    rank_hdf5_file = tables.open_file(os.path.join(work_dir, 'initial_rank_pearson.hdf5'), mode='r')
    initial_rank = rank_hdf5_file.root.initial_rank
    v_hdf5_file = tables.open_file(os.path.join(work_dir, f'V_{num}.hdf5'), mode='w')
    V = v_hdf5_file.create_carray(v_hdf5_file.root, 'V', tables.Float32Atom(), shape=(end-start, ALL_NUM), filters=tables.Filters())

    for i in tqdm(range(start, end), desc='calculate V'):
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        # weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        # V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
        V[i-start, k_reciprocal_expansion_index] = original_dist[i,k_reciprocal_expansion_index]
    
    v_hdf5_file.close()
    rank_hdf5_file.close()
    dist_hdf5_file.close()

#calc_V(4,320000,400000)

def cat_V():
    v_hdf5_file = tables.open_file(os.path.join(work_dir, f'V.hdf5'), mode='w')
    V = v_hdf5_file.create_earray(v_hdf5_file.root, 'V', tables.Float32Atom(), shape=(0, ALL_NUM), filters=tables.Filters(), expectedrows=ALL_NUM)
    for i in range(10):
        vn_hdf5_file = tables.open_file(os.path.join(work_dir, f'V_{i}.hdf5'), mode='r')
        Vn = vn_hdf5_file.root.V
        for j in tqdm(range(0, Vn.shape[0]), desc=f'cat V:{i}'):
            V.append(Vn[j: j+1])
        vn_hdf5_file.close()
    v_hdf5_file.close()

#cat_V()

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

BS = 200
for i in range(0, 4):
    calc_jaccard(i, i*BS, (i+1)*BS)

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

query_names = torch.load('/nfs3-p2/zsxm/naic/preliminary/test_B/query_names.pt')
gallery_names = torch.load('/nfs3-p2/zsxm/naic/preliminary/test_B/gallery_names.pt')

def calc_result(num, start, end):
    batch_size = 16
    jaccard_dir = os.path.join(work_dir, f'jaccard_dist_{num}.pt')
    origin_dir = os.path.join(work_dir, 'original_dist_pearson.hdf5')
    fdataset = LoadDist(jaccard_dir, origin_dir, num, BS)
    floader = DataLoader(fdataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    res_dict = {}
    for index, fdist in tqdm(floader, desc=f'calc result {num}-{start}-{end}'):
        idx = torch.argsort(fdist, dim=-1, descending=True)
        for i in len(idx):
            name = query_names[index[i]]
            query_res = []
            for j in range(100):
                query_res.append(gallery_names[idx[i, j]])
            res_dict[name] = query_res
    
    torch.save(res_dict, os.path.join(work_dir, f'res_{num}.pt'))

def cat_result():
    res = {}
    for i in range():
        resi = torch.load(os.path.join(work_dir, f'res_{i}.pt'))
        for k, v in resi.items():
            res[k] = v

    with open('/home/zsxm/pythonWorkspace/NAIC/result_jaccard_b.json', 'w') as f:
        json.dump(res, f)
    