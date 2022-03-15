from project.extract import extract
from project.compress import compress
from project.reconstruct import reconstruct
from project.reid import reid

import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# extract()
# os.rename('feature', 'query_feature')

# for byte_rate in ['64', '128', '256']:
#     compress(byte_rate)

# for byte_rate in ['64', '128', '256']:
#     reconstruct(byte_rate)

# for byte_rate in ['64', '128', '256']:
#     reid(byte_rate)

# class L2ReconstructionLoss(nn.Module):
#     def __init__(self):
#         super(L2ReconstructionLoss, self).__init__()

#     def forward(self, reconstructed, origin):
#         assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
#         return torch.linalg.norm(reconstructed-origin, dim=1, ord=2).mean()

# file_list = sorted(os.listdir('query_feature'))
# feature_list = []
# for file in file_list:
#     feature_list.append(np.fromfile(os.path.join('query_feature', file), dtype='<f4')[:128])
# features = torch.from_numpy(np.stack(feature_list, axis=0))
# del feature_list
# for byte_rate in ['64', '128', '256']:
#     reconstruct_feature_list = []
#     for file in file_list:
#         reconstruct_feature_list.append(np.fromfile(os.path.join('reconstructed_query_feature', byte_rate, file), dtype='<f4')[:128])
#     reconstruct_features = torch.from_numpy(np.stack(reconstruct_feature_list, axis=0))
#     del reconstruct_feature_list
#     print(f'byte_rate {byte_rate} L2 loss:', L2ReconstructionLoss()(reconstruct_features, features))
# byte_rate 64 L2 loss: tensor(0.1034)
# byte_rate 128 L2 loss: tensor(0.1034)
# byte_rate 256 L2 loss: tensor(5.8545e-05)

def calc_acc_reid(query, gallary, labels):
    device = torch.device('cuda:1')
    query = F.normalize(query.to(device), dim=1)
    gallary = F.normalize(gallary.to(device), dim=1)
    dists = torch.mm(query, gallary.T)
    ranks = torch.argsort(dists, dim=1, descending=True).cpu().numpy()
    del dists, query, gallary
    labels = labels.numpy()

    acc1, mAP = 0, 0
    for i, rank in enumerate(ranks):
        ap = 0
        rank = rank[rank!=i]
        label = labels[i]
        rank_label = np.take_along_axis(labels, rank, axis=0)
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
    
    acc1 /= ranks.shape[0]
    mAP /= ranks.shape[0]
    ACC_reid = (acc1 + mAP) / 2
    return ACC_reid, acc1, mAP


with open('/nfs3-p2/zsxm/naic/rematch/train/val_list.txt', 'r') as f:
    lines = f.readlines()
file_to_label = {}
for line in lines:
    fname, label = line.split()[:2]
    fname = fname.split('.')[0]
    label = int(label)
    file_to_label[fname] = label
file_list = sorted(os.listdir('query_feature'))
feature_list = []
label_list = []
for file in file_list:
    feature_list.append(np.fromfile(os.path.join('query_feature', file), dtype='<f4')[:128])
    label_list.append(file_to_label[file.split('.')[0]])
labels = torch.tensor(label_list, dtype=torch.long)
features = torch.from_numpy(np.stack(feature_list, axis=0))
del feature_list, label_list
ACC_reid, acc1, mAP = calc_acc_reid(features, features, labels)
print('features x features:', ACC_reid, acc1, mAP)
for byte_rate in ['64', '128', '256']:
    reconstruct_feature_list = []
    for file in file_list:
        reconstruct_feature_list.append(np.fromfile(os.path.join('reconstructed_query_feature', byte_rate, file), dtype='<f4')[:128])
    reconstruct_features = torch.from_numpy(np.stack(reconstruct_feature_list, axis=0))
    del reconstruct_feature_list
    ACC_reid, acc1, mAP = calc_acc_reid(reconstruct_features, features, labels)
    print(f'{byte_rate}_reco x features:', ACC_reid, acc1, mAP)