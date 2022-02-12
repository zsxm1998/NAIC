import os
import numpy as np
import torch
import tables
from tqdm import tqdm
from torch.utils.data import Dataset

ALL_NUM = 448794
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
    for i in range(6):
        vn_hdf5_file = tables.open_file(os.path.join(work_dir, f'V_{i}.hdf5'), mode='r')
        Vn = vn_hdf5_file.root.V
        for j in tqdm(range(0, Vn.shape[0])):
            V.append(Vn[j: j+1])
        vn_hdf5_file.close()
    v_hdf5_file.close()

#cat_V()

class LoadV(Dataset):
    def __init__(self, dir):
        self.v_hdf5_file = tables.open_file(os.path.join(work_dir, f'V.hdf5'), mode='r')
        self.V = self.v_hdf5_file.root.V

    def __len__(self):
        return self.V.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.V[index])

    def __del__(self):


def calc_jaccard(num, start, end):
    v_hdf5_file = tables.open_file(os.path.join(work_dir, f'V.hdf5'), mode='r')
    V = v_hdf5_file.root.V
    for i in tqdm(range(start, end), desc='calculate jaccard distance'):
        probe = V[]