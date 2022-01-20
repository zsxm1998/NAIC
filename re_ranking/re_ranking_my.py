#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

dis_path = '/nfs3-p2/zsxm/naic/preliminary/test_A/dis'
if not os.path.exists(dis_path):
    os.makedirs(dis_path)

def get_vector(name, i, batch):
    if isinstance(i, (int, np.int64, np.int32)):
        i = int(i)
        j = i // batch
        k = i % batch
        temp = np.load(os.path.join(dis_path, f'{name}-{j}.npy'))
        res = temp[k].copy()
        return res
    else:
        assert isinstance(i, list) or i.ndim == 1, f'type of i: {type(i)}, content of i: {i}'
        i = i if isinstance(i, list) else i.tolist()
        res = []
        above_j, temp = -1, 0
        for idx in i:
            j = idx // batch
            k = idx % batch
            if above_j != j:
                del temp
                temp = np.load(os.path.join(dis_path, f'{name}-{j}.npy'))
                above_j = j
            res.append(temp[k].copy())
        return np.stack(res, axis=0)

def get_col(name, i, batch, length):
    tot = (length+batch-1) // batch + 1
    if isinstance(i, int):
        res = []
        for j in range(tot):
            temp = np.load(os.path.join(dis_path, f'{name}-{j}.npy'))
            res.append(temp[:, i].copy())
            #del temp
        return np.concatenate(res)
    else:
        raise NotImplementedError('get_col i not int is not implemented.') 

def calc_dist(probFea, galFea, k1, batch, calc_flag=True):
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0] 
    gallery_num = all_num   
    if calc_flag:
        feat = np.append(probFea,galFea,axis = 0)
        feat = feat.astype(np.float32)
        device = torch.device('cuda:0')
        feat = torch.from_numpy(feat)
        for t, i in enumerate(tqdm(range(0, feat.shape[0], batch), desc='computing original distance')):
            dist_res_list = []
            for j in range(0, feat.shape[0], batch):
                a, b = feat[i:i+batch, None].to(device), feat[None, j:j+batch].to(device)
                dist_res = (a-b).pow_(2).sum(dim=-1)
                dist_res /= torch.max(dist_res, dim=1, keepdim=True)[0] #np.power(cdist(feat[i: i+batch], feat), 2).astype(np.float32)
                dist_res_list.append(dist_res.cpu()) #1. * dist_res / np.max(dist_res, axis=1, keepdims=True)
            dist_res = torch.cat(dist_res_list, dim=1).numpy()
            np.save(os.path.join(dis_path, f'original_dist-{t}.npy'), dist_res)
            rank_res = np.argpartition(dist_res, range(1,k1+1))[:, :k1+1]
            np.save(os.path.join(dis_path, f'initial_rank-{t}.npy'), rank_res)
            # V_res = np.zeros_like(dist_res).astype(np.float32)
            # np.save(os.path.join(dis_path, f'V_{t}.npy'), V_res)
    return query_num, gallery_num, all_num

class SaveV():
    def __init__(self, batch, name):
        self.batch = batch
        self.mem = []
        self.count = 0
        self.name = name
    
    def save_v(self, v):
        self.mem.append(v)
        if len(v) == self.batch:
            V_res = np.stack(self.mem, axis=0)
            np.save(os.path.join(dis_path, f'{self.name}-{self.count}.npy'), V_res)
            self.count += 1
            self.mem.clear()

    def clear(self, name='V'):
        if len(self.mem) != 0:
            V_res = np.stack(self.mem, axis=0)
            np.save(os.path.join(dis_path, f'{self.name}-{self.count}.npy'), V_res)
        self.mem.clear()
        self.count = 0
        self.name = name

def re_ranking(probFea, galFea, k1, k2, lambda_value, batch=1000):
    query_num, gallery_num, all_num = calc_dist(probFea, galFea, k1, batch, False)

    print('starting re_ranking')
    sv = SaveV(batch, 'V')
    for i in tqdm(range(all_num), desc='k-reciprocal neighbors'):
        forward_k_neigh_index = get_vector('initial_rank', i, batch)[:k1+1] #initial_rank[i,:k1+1]
        backward_k_neigh_index = get_vector('initial_rank', forward_k_neigh_index, batch)[:, :k1+1] #initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        del backward_k_neigh_index
        k_reciprocal_index = forward_k_neigh_index[fi].copy()
        del forward_k_neigh_index
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = get_vector('initial_rank', candidate, batch)[:, :int(np.around(k1/2))+1]#initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = get_vector('initial_rank', candidate_forward_k_neigh_index, batch)[:, :int(np.around(k1/2))+1]#initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            del fi_candidate, candidate_backward_k_neigh_index, candidate_forward_k_neigh_index
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-get_vector('original_dist', i, batch)[k_reciprocal_expansion_index])#np.exp(-original_dist[i,k_reciprocal_expansion_index])
        v_temp = np.zeros(all_num, dtype=np.float32)
        v_temp[k_reciprocal_expansion_index] = weight/np.sum(weight)
        sv.save_v(v_temp)#V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    sv.clear('V_qe')

    #original_dist = original_dist[:query_num,]    
    if k2 != 1:
        #V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            v_qe_temp = np.mean(get_vector('V', get_vector('initial_rank', i, batch)[:k2], batch), axis=0)
            sv.save_v(v_qe_temp) #V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
    sv.clear('jaccard_dist')
    
    invIndex = []
    for i in range(gallery_num):
        v_temp = get_col('V_qe', i, batch, all_num)
        invIndex.append(np.where(v_temp != 0)[0])
    
    #jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(get_vector('V_qe', i, batch) != 0)[0] #np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]] + np.minimum(get_vector('V_qe', i, batch)[indNonZero[j]], get_vector('V_qe', indImages[j], batch)[:, indNonZero[j]]) #np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        temp_jaccard_dist = 1-temp_min/(2-temp_min) #jaccard_dist[i] = 1-temp_min/(2-temp_min)
        sv.save_v(temp_jaccard_dist)
    sv.clear('final_dist')
    
    #final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    final_dist = np.zeros((query_num, all_num-query_num))
    for t, i in enumerate(range(0, query_num, batch)):
        original_dist = np.load(os.path.join(dis_path, f'original_dist-{t}.npy'))
        jaccard_dist = np.load(os.path.join(dis_path, f'jaccard_dist-{t}.npy'))
        final_dist[i, i+batch] = (jaccard_dist*(1-lambda_value) + original_dist*lambda_value)[:, query_num:]

    return final_dist
