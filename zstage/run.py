from project.extract import extract
from project.compress import compress
from project.reconstruct import reconstruct
from project.reid import reid

import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root = '/nfs3-p1/zsxm/naic/rematch/test'

# extract(root=root)
# os.rename(os.path.join(root, 'feature'), os.path.join(root, 'query_feature'))

# for byte_rate in ['64', '128', '256']:
#     compress(byte_rate, root=root)

# for byte_rate in ['64', '128', '256']:
#     reconstruct(byte_rate, root=root)

# class L2ReconstructionLoss(nn.Module):
#     def __init__(self):
#         super(L2ReconstructionLoss, self).__init__()

#     def forward(self, reconstructed, origin):
#         assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
#         return torch.linalg.norm(reconstructed-origin, dim=1, ord=2).mean()

# file_list = sorted(os.listdir(os.path.join(root, 'query_feature')))
# feature_list = []
# for file in file_list:
#     feature_list.append(np.fromfile(os.path.join(root, 'query_feature', file), dtype='<f4')[:128])
# features = torch.from_numpy(np.stack(feature_list, axis=0))
# del feature_list
# for byte_rate in ['64', '128', '256']:
#     reconstruct_feature_list = []
#     for file in file_list:
#         reconstruct_feature_list.append(np.fromfile(os.path.join(root, 'reconstructed_query_feature', byte_rate, file), dtype='<f4')[:128])
#     reconstruct_features = torch.from_numpy(np.stack(reconstruct_feature_list, axis=0))
#     del reconstruct_feature_list
#     print(f'byte_rate {byte_rate} L2 loss:', L2ReconstructionLoss()(reconstruct_features, features).item())
# byte_rate 64 L2 loss: tensor(0.1429)
# byte_rate 128 L2 loss: tensor(0.1429)
# byte_rate 256 L2 loss: tensor(0.0002)
# huffman误差
# byte_rate 64 L2 loss: tensor(0.6154)
# byte_rate 128 L2 loss: tensor(0.4219)
# byte_rate 256 L2 loss: tensor(5.8545e-05)

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


# with open('/nfs3-p2/zsxm/naic/rematch/train/val_list.txt', 'r') as f:
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
#     feature_list.append(np.fromfile(os.path.join(root, 'query_feature', file), dtype='<f4')[:128])
#     label_list.append(file_to_label[file.split('.')[0]])
# labels = torch.tensor(label_list, dtype=torch.long)
# features = torch.from_numpy(np.stack(feature_list, axis=0))
# del feature_list, label_list
# ACC_reid, acc1, mAP = calc_acc_reid(features, features, labels)
# print('features x features:', ACC_reid, acc1, mAP)
# for byte_rate in ['64', '128', '256']:
#     reconstruct_feature_list = []
#     for file in file_list:
#         reconstruct_feature_list.append(np.fromfile(os.path.join(root, 'reconstructed_query_feature', byte_rate, file), dtype='<f4')[:128])
#     reconstruct_features = torch.from_numpy(np.stack(reconstruct_feature_list, axis=0))
#     del reconstruct_feature_list
#     ACC_reid, acc1, mAP = calc_acc_reid(reconstruct_features, reconstruct_features, labels)
#     print(f'{byte_rate}_reco x {byte_rate}_reco:', ACC_reid, acc1, mAP)
#     ACC_reid, acc1, mAP = calc_acc_reid(reconstruct_features, features, labels)
#     print(f'{byte_rate}_reco x features:', ACC_reid, acc1, mAP)
# features x features: 0.9238112850858411 0.9532313335585695 0.8943912366131126
# 64_reco x 64_reco: 0.8914029844633248 0.9276992132824944 0.8551067556441552
# 64_reco x features: 0.8678425129586782 0.905207780298277 0.8304772456190793
# 128_reco x 128_reco: 0.8914009035879379 0.9276992132824944 0.8551025938933814
# 128_reco x features: 0.8678170211641028 0.9051595154206284 0.8304745269075771
# 256_reco x 256_reco: 0.9238110803392543 0.9532313335585695 0.894390827119939
# 256_reco x features: 0.9238110821898237 0.9532313335585695 0.8943908308210778

for byte_rate in ['64', '128', '256']:
    reid(byte_rate, root=root, method='pearson')

with open('/nfs3-p2/zsxm/naic/rematch/train/val_list.txt', 'r') as f:
    lines = f.readlines()
file_to_label = {}
for line in lines:
    fname, label = line.split()[:2]
    fname = fname.split('.')[0]+'.png'
    label = int(label)
    file_to_label[fname] = label
for byte_rate in ['64', '128', '256']:
    with open(os.path.join(root, f'reid_results/{byte_rate}.json'), 'r') as jf:
        res = json.load(jf)
    acc1, mAP = 0, 0
    for k, v in res.items():
        ap = 0
        label = file_to_label[k]
        rank_label = np.array([file_to_label[fn] for fn in v if fn != k])
        if rank_label[0] == label:
            acc1 += 1 
        correct_rank_idx = np.argwhere(rank_label==label).flatten()
        correct_rank_idx = correct_rank_idx[correct_rank_idx<100]
        n_correct = len(correct_rank_idx)
        if n_correct > 0:
            d_recall = 1 / n_correct
            for j in range(n_correct):
                precision = (j+1) / (correct_rank_idx[j]+1)
                ap += d_recall * precision
        mAP += ap
    acc1 /= len(res)
    mAP /= len(res)
    ACC_reid = (acc1 + mAP) / 2
    print(f'{byte_rate}', ACC_reid, acc1, mAP)
# cosine
# 64 0.9327406865564398 0.9631738983541677 0.9023074747587121
# 128 0.9327396509865842 0.9631738983541677 0.9023054036190007
# 256 0.9475883626951143 0.974660939234519 0.9205157861557094
# pearson
# 64 0.9329493515842285 0.9633669578647618 0.9025317453036953
# 128 0.932945212276817 0.9633669578647618 0.9025234666888723
# 256 0.9476828381121662 0.9747574689898161 0.9206082072345165
# l2
# 64 0.934478070005946 0.9640426661518413 0.9049134738600506
# 128 0.9344474520495178 0.9639944012741928 0.9049005028248429
# 256 0.9484113810523266 0.9749022636227617 0.9219204984818915
# rerank_cosine
# 64 0.9462889661213735 0.965007963704812 0.9275699685379349
# 128 0.9462911015538078 0.965007963704812 0.9275742394028037
# 256 0.9539670240787324 0.974660939234519 0.9332731089229456
# rerank_pearson
# 64 0.9464147995663565 0.9651527583377576 0.9276768407949555
# 128 0.9464163297041263 0.9651527583377576 0.9276799010704949
# 256 0.954103446930298 0.9747574689898161 0.9334494248707801
# rerank_l2
# 64 0.9481470141309696 0.966359380278971 0.9299346479829683
# 128 0.9481091489499056 0.9663111154013224 0.9299071824984887
# 256 0.9548656637771802 0.9749022636227617 0.9348290639315986