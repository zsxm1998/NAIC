from project_huffman.extract import extract
from project_huffman.compress import compress
from project_huffman.reconstruct import reconstruct
from project_huffman.reid import reid

import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import json
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
root = '/nfs3-p2/zsxm/naic/rematch/test_huffman'
DIM_NUM = 180

t1 = time.time()
extract(root=root)
print('time', time.time()-t1, 's')
os.rename(os.path.join(root, 'feature'), os.path.join(root, 'query_feature'))



# for byte_rate in ['64', '128', '256']:
#     t1 = time.time()
#     compress(byte_rate, root=root)
#     print('time', time.time()-t1, 's')

# for byte_rate in ['64', '128', '256']:
#     t1 = time.time()
#     reconstruct(byte_rate, root=root)
#     print('time', time.time()-t1, 's')

# class L2ReconstructionLoss(nn.Module):
#     def __init__(self):
#         super(L2ReconstructionLoss, self).__init__()

#     def forward(self, reconstructed, origin):
#         assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
#         return torch.linalg.norm(reconstructed-origin, dim=1, ord=2).mean()

# file_list = sorted(os.listdir(os.path.join(root, 'query_feature')))
# feature_list = []
# for file in file_list:
#     feature_list.append(np.fromfile(os.path.join(root, 'query_feature', file), dtype='<f4')[:DIM_NUM])
# features = torch.from_numpy(np.stack(feature_list, axis=0))
# del feature_list
# for byte_rate in ['64', '128', '256']:
#     reconstruct_feature_list = []
#     for file in file_list:
#         reconstruct_feature_list.append(np.fromfile(os.path.join(root, 'reconstructed_query_feature', byte_rate, file), dtype='<f4')[:DIM_NUM])
#     reconstruct_features = torch.from_numpy(np.stack(reconstruct_feature_list, axis=0))
#     del reconstruct_feature_list
#     print(f'byte_rate {byte_rate} L2 loss:', L2ReconstructionLoss()(reconstruct_features, features).item())



# for byte_rate in ['64', '128', '256']:
#     t1 = time.time()
#     reid(byte_rate, root=root, method='cosine', after=True)
#     print('time', time.time()-t1, 's')

# with open(os.path.join(root, 'val_list.txt'), 'r') as f:
#     lines = f.readlines()
# file_to_label = {}
# for line in lines:
#     fname, label = line.split()[:2]
#     fname = fname.split('.')[0]+'.png'
#     label = int(label)
#     file_to_label[fname] = label
# for byte_rate in ['64', '128', '256']:
#     with open(os.path.join(root, f'reid_results/{byte_rate}.json'), 'r') as jf:
#         res = json.load(jf)
#     acc1, mAP = 0, 0
#     for k, v in res.items():
#         ap = 0
#         label = file_to_label[k]
#         rank_label = np.array([file_to_label[fn] for fn in v if fn != k])
#         if rank_label[0] == label:
#             acc1 += 1 
#         correct_rank_idx = np.argwhere(rank_label==label).flatten()
#         correct_rank_idx = correct_rank_idx[correct_rank_idx<100]
#         n_correct = len(correct_rank_idx)
#         if n_correct > 0:
#             d_recall = 1 / n_correct
#             for j in range(n_correct):
#                 precision = (j+1) / (correct_rank_idx[j]+1)
#                 ap += d_recall * precision
#         mAP += ap
#     acc1 /= len(res)
#     mAP /= len(res)
#     ACC_reid = (acc1 + mAP) / 2
#     print(f'{byte_rate}', ACC_reid, acc1, mAP)














# 不通过reid.py测结果---------------------------------------------------------------------------------------
# def calc_acc_reid(query, gallary, labels):
#     device = torch.device('cuda:0')
#     query = F.normalize(query.to(device), dim=1)
#     gallary = F.normalize(gallary.to(device), dim=1)
#     dists = torch.mm(query, gallary.T)
#     ranks = torch.argsort(dists, dim=1, descending=True).cpu().numpy()
#     del dists, query, gallary
#     labels = labels.numpy()

#     acc1, mAP = 0, 0
#     for i, rank in enumerate(ranks):
#         ap = 0
#         rank = rank[rank!=i]
#         label = labels[i]
#         rank_label = np.take_along_axis(labels, rank, axis=0)
#         if rank_label[0] == label:
#             acc1 += 1 
#         correct_rank_idx = np.argwhere(rank_label==label).flatten()
#         correct_rank_idx = correct_rank_idx[correct_rank_idx<100]
#         n_correct = len(correct_rank_idx)
#         if n_correct > 0:
#             d_recall = 1 / n_correct
#             for j in range(n_correct):
#                 precision = (j+1) / (correct_rank_idx[j]+1)
#                 ap += d_recall * precision
#         mAP += ap
    
#     acc1 /= ranks.shape[0]
#     mAP /= ranks.shape[0]
#     ACC_reid = (acc1 + mAP) / 2
#     return ACC_reid, acc1, mAP


# with open(os.path.join(root, 'val_list.txt'), 'r') as f:
#     lines = f.readlines()
# file_to_label = {}
# for line in lines:
#     fname, label = line.split()[:2]
#     fname = fname.split('.')[0]
#     label = int(label)
#     file_to_label[fname] = label
# file_list = sorted(os.listdir(os.path.join(root, 'query_feature')))
# feature_list = []
# label_list = []
# for file in file_list:
#     feature_list.append(np.fromfile(os.path.join(root, 'query_feature', file), dtype='<f4')[:DIM_NUM])
#     label_list.append(file_to_label[file.split('.')[0]])
# labels = torch.tensor(label_list, dtype=torch.long)
# features = torch.from_numpy(np.stack(feature_list, axis=0))
# del feature_list, label_list
# ACC_reid, acc1, mAP = calc_acc_reid(features, features, labels)
# print('features x features:', ACC_reid, acc1, mAP)
# for byte_rate in ['64', '128', '256']:
#     reconstruct_feature_list = []
#     for file in file_list:
#         reconstruct_feature_list.append(np.fromfile(os.path.join(root, 'reconstructed_query_feature', byte_rate, file), dtype='<f4')[:DIM_NUM])
#     reconstruct_features = torch.from_numpy(np.stack(reconstruct_feature_list, axis=0))
#     del reconstruct_feature_list
#     ACC_reid, acc1, mAP = calc_acc_reid(reconstruct_features, reconstruct_features, labels)
#     print(f'{byte_rate}_reco x {byte_rate}_reco:', ACC_reid, acc1, mAP)
#     ACC_reid, acc1, mAP = calc_acc_reid(reconstruct_features, features, labels)
#     print(f'{byte_rate}_reco x features:', ACC_reid, acc1, mAP)
#---------------------------------------------------------------------------------------