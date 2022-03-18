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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
root = '/nfs3-p1/zsxm/naic/rematch/test'

# extract(root=root)
# os.rename(os.path.join(root, 'feature'), os.path.join(root, 'query_feature'))

t1 = time.time()
for byte_rate in ['64', '128', '256']:
    compress(byte_rate, root=root)

for byte_rate in ['64', '128', '256']:
    reconstruct(byte_rate, root=root)
print(time.time()-t1)


class L2ReconstructionLoss(nn.Module):
    def __init__(self):
        super(L2ReconstructionLoss, self).__init__()

    def forward(self, reconstructed, origin):
        assert reconstructed.shape == origin.shape, f'reconstructed.shape({reconstructed.shape}) should be equal to origin.shape({origin.shape})'
        return torch.linalg.norm(reconstructed-origin, dim=1, ord=2).mean()

file_list = sorted(os.listdir(os.path.join(root, 'query_feature')))
feature_list = []
for file in file_list:
    feature_list.append(np.fromfile(os.path.join(root, 'query_feature', file), dtype='<f4')[:128])
features = torch.from_numpy(np.stack(feature_list, axis=0))
del feature_list
for byte_rate in ['64', '128', '256']:
    reconstruct_feature_list = []
    for file in file_list:
        reconstruct_feature_list.append(np.fromfile(os.path.join(root, 'reconstructed_query_feature', byte_rate, file), dtype='<f4')[:128])
    reconstruct_features = torch.from_numpy(np.stack(reconstruct_feature_list, axis=0))
    del reconstruct_feature_list
    print(f'byte_rate {byte_rate} L2 loss:', L2ReconstructionLoss()(reconstruct_features, features).item())
# huffman 470.00023436546326
# byte_rate 64 L2 loss: 0.4872712790966034
# byte_rate 128 L2 loss: 0.10426277667284012
# byte_rate 256 L2 loss: 0.00046082952758297324


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
# # features x features: 0.9542052946353399 0.9783290699358077 0.9300815193348722
# # 64_reco x 64_reco: 0.9450051197654893 0.9722476953520923 0.9177625441788863
# # 64_reco x features: 0.9426398744038782 0.9694965973261258 0.9157831514816306
# # 128_reco x 128_reco: 0.9450067343835689 0.9722476953520923 0.9177657734150455
# # 128_reco x features: 0.9426378973257936 0.9694965973261258 0.9157791973254615
# # 256_reco x 256_reco: 0.9542064843072109 0.9783290699358077 0.9300838986786141
# # 256_reco x features: 0.9542045784552906 0.9783290699358077 0.9300800869747736

# for byte_rate in ['64', '128', '256']:
#     reid(byte_rate, root=root, method='rerank_l2')

# with open('/nfs3-p2/zsxm/naic/rematch/train/val_list.txt', 'r') as f:
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
# 64 0.9807328575112684 0.9901539649596989 0.9713117500628378
# 128 0.9827711974760378 0.9923741493315315 0.9731682456205443
# 256 0.9829542702954991 0.992518943964477 0.9733895966265211













# pearson
# 64 0.9426747695268491 0.9694965973261258 0.9158529417275725
# 128 0.9426741014818318 0.9694965973261258 0.9158516056375379
# 256 0.9539986869771379 0.9777981562816739 0.9301992176726018
# l2
# 64 0.9448386829009084 0.9708962787779333 0.9187810870238836
# 128 0.944841577505252 0.9708962787779333 0.9187868762325707
# 256 0.9550639590062662 0.9782808050581592 0.9318471129543731
# rerank_l2 [0.02, 0.56, 0.24, 0.1, 0.02, 0.01, 0.02, 0.02, 0.02]
# 64 0.9557999786046092 0.9720063709638496 0.9395935862453687
# 128 0.9558015052555274 0.9720063709638496 0.9395966395472053
# 256 0.9611230327047765 0.9782808050581592 0.9439652603513938
# rerank_l2 [0.0, 0.7, 0.3]
# 64 0.9542752608247811 0.9710410734108789 0.9375094482386832
# 128 0.9542754616879229 0.9710410734108789 0.9375098499649669
# 256 0.9594750598079109 0.9782808050581592 0.9406693145576626
# rerank_l2 [0.3, 0.6, 0.1]
# 64 0.9526465207713437 0.9733577875380086 0.9319352540046787
# 128 0.952645234932014 0.9733577875380086 0.9319326823260193
# 256 0.9569711422417269 0.9782808050581592 0.9356614794252947