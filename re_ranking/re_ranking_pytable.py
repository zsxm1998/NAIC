import os
import numpy as np
import tables
from tqdm import tqdm

dis_path = '/nfs3-p2/zsxm/naic/preliminary/test_A/dis'

def re_ranking(probFea,galFea,k1,k2,lambda_value, MemorySave = False, Minibatch = 2000):
    query_num = probFea #probFea.shape[0]
    all_num = query_num+galFea #query_num + galFea.shape[0]
    gallery_num = all_num

    data_hdf5_path = os.path.join(dis_path, 'reranking.hdf5')
    data_hdf5_file = tables.open_file(data_hdf5_path, mode='r')
    original_dist = data_hdf5_file.root.original_dist
    initial_rank = data_hdf5_file.root.initial_rank
    temp_hdf5_path = os.path.join(dis_path, 'reranking_temp.hdf5')
    temp_hdf5_file = tables.open_file(temp_hdf5_path, mode='w')
    filters = tables.Filters()
    V = temp_hdf5_file.create_carray(temp_hdf5_file.root, 'V', tables.Float32Atom(), shape=(all_num, all_num), filters=filters)

    try:
        for i in tqdm(range(all_num), desc='starting re_ranking'):
            # k-reciprocal neighbors
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
            weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
            V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
        
        original_dist = original_dist[:query_num,]    
        if k2 != 1:
            V_qe = temp_hdf5_file.create_carray(temp_hdf5_file.root, 'V_qe', tables.Float32Atom(), shape=(all_num, all_num), filters=filters) #V_qe = np.zeros_like(V,dtype=np.float32)
            for i in tqdm(range(all_num), desc='calculate, V_qe'):
                V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
            V = V_qe
            del V_qe
        del initial_rank

        invIndex = []
        for i in tqdm(range(gallery_num), desc='calculate invIndex'):
            invIndex.append(np.where(V[:,i] != 0)[0])
        
        jaccard_dist = np.zeros((query_num, all_num), dtype = np.float32)
        for i in tqdm(range(query_num), desc='calculate jaccard_dist'):
            temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
            indNonZero = np.where(V[i,:] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
            jaccard_dist[i] = 1-temp_min/(2-temp_min)
        
        final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
        del original_dist
        del V
        del jaccard_dist
        final_dist = final_dist[:query_num,query_num:]

        data_hdf5_file.close()
        temp_hdf5_file.close()
    except:
        data_hdf5_file.close()
        temp_hdf5_file.close()
    return final_dist