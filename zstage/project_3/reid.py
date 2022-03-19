import os
import json
import gc

import numpy as np
import torch
import torch.nn.functional as F

from .ae import AEModel

DIM_NUM = 1024
BATCH_SIZE = 512
L2_BATCH_SIZE = 6
WEIGHT = [0.02, 0.56, 0.24, 0.1, 0.02, 0.01, 0.02, 0.02, 0.02]

def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')[:DIM_NUM]

def l2_dist(q, k):
    return torch.linalg.norm(q.unsqueeze(-2) - k.unsqueeze(0), dim=-1, ord=2)

def cosine_similarity(q, k):
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    return torch.mm(q, k.T)

def pearson_similarity(q, k):
    q = q - q.mean(dim=-1, keepdim=True)
    k = k - k.mean(dim=-1, keepdim=True)
    return cosine_similarity(q, k)

@torch.no_grad()
def reid(bytes_rate, root='', method='rerank_l2', after=True):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    reconstructed_query_fea_dir = os.path.join(root, 'reconstructed_query_feature/{}'.format(bytes_rate))
    gallery_fea_dir = 'gallery_feature' if root == '' else os.path.join(root, 'query_feature')
    reid_results_path = os.path.join(root, 'reid_results/{}.json'.format(bytes_rate))
    os.makedirs(os.path.dirname(reid_results_path), exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    query_names = sorted(os.listdir(reconstructed_query_fea_dir))
    gallery_names = sorted(os.listdir(gallery_fea_dir))
    query_num = len(query_names)
    gallery_num = len(gallery_names)
    assert(query_num != 0 and gallery_num != 0)
    top_num = min(100, gallery_num)
    reconstructed_query_fea_list = []
    gallery_fea_list = []
    for query_name in query_names:
        reconstructed_query_fea_list.append(
            read_feature_file(os.path.join(reconstructed_query_fea_dir, query_name))
        )
    for gallery_name in gallery_names:
        gallery_fea_list.append(
            read_feature_file(os.path.join(gallery_fea_dir, gallery_name))
        )
    reconstructed_query_fea_all = torch.from_numpy(np.stack(reconstructed_query_fea_list, axis=0)).to(device)
    gallery_fea_all = torch.from_numpy(np.stack(gallery_fea_list, axis=0)).to(device)
    del reconstructed_query_fea_list, gallery_fea_list
    gc.collect()

    ae_net = AEModel('efficientnet_b4(num_classes={})', extractor_out_dim=DIM_NUM, compress_dim=32)
    ae_net.load_param(os.path.join(root, f'project/Net_best.pth'))
    ae_net.to(device)
    ae_net.eval()
    reco_gallery_list = []
    for i in range(0, gallery_num, BATCH_SIZE):
        j = min(i+BATCH_SIZE, gallery_num)
        reco_gallery_list.append(ae_net.ae(gallery_fea_all[i:j], bytes_rate))
    del gallery_fea_all
    torch.cuda.empty_cache()
    gallery_fea_all = torch.cat(reco_gallery_list, dim=0).to(device)
    del reco_gallery_list

    if after:
        bn_query_list = []
        for i in range(0, query_num, BATCH_SIZE):
            j = min(i+BATCH_SIZE, query_num)
            bn_query_list.append(ae_net.bn(reconstructed_query_fea_all[i:j], bytes_rate))
        del reconstructed_query_fea_all
        torch.cuda.empty_cache()
        reconstructed_query_fea_all = torch.cat(bn_query_list, dim=0).to(device)
        del bn_query_list
        bn_gallery_list = []
        for i in range(0, gallery_num, BATCH_SIZE):
            j = min(i+BATCH_SIZE, gallery_num)
            bn_gallery_list.append(ae_net.bn(gallery_fea_all[i:j], bytes_rate))
        del gallery_fea_all
        torch.cuda.empty_cache()
        gallery_fea_all = torch.cat(bn_gallery_list, dim=0).to(device)
        del bn_gallery_list
    del ae_net
    torch.cuda.empty_cache()

    query_names = [name.rsplit('.', 1)[0] + '.png' for name in query_names]
    gallery_names_array = np.array(list(map(lambda _: _.rsplit('.', 1)[0] + '.png', gallery_names)))
    del gallery_names

    dist_func = {'cosine': cosine_similarity,
                 'pearson': pearson_similarity,
                 'l2': l2_dist,
                 'rerank_cosine': cosine_similarity,
                 'rerank_pearson': pearson_similarity,
                 'rerank_l2': l2_dist,
                }
    batch_size = L2_BATCH_SIZE if 'l2' in method else BATCH_SIZE
    result_dict = {}
    for i in range(0, len(query_names), batch_size):
        j = min(i+batch_size, len(query_names))
        query_idx = torch.arange(i, j)
        dist = dist_func[method](reconstructed_query_fea_all[query_idx], gallery_fea_all)
        indexes = torch.argsort(dist, dim=-1, descending = 'l2' not in method)[:, :top_num].cpu()
        del dist
        torch.cuda.empty_cache()

        if 'rerank' not in method:
            indexes = indexes.numpy()
        else:
            probe = reconstructed_query_fea_all[query_idx] * WEIGHT[0]
            for w in range(len(WEIGHT) - 1):
                probe += WEIGHT[w+1] * gallery_fea_all[indexes[:, w]]
            dist = dist_func[method](probe, gallery_fea_all)
            indexes = torch.argsort(dist, dim=-1, descending = 'l2' not in method)[:, :top_num].cpu().numpy()
            del dist
        
        for s, k in enumerate(range(i, j)):
            result_dict[query_names[k]] = gallery_names_array[indexes[s]].tolist()

        del indexes
        gc.collect()
        torch.cuda.empty_cache()

    with open(reid_results_path, 'w', encoding='UTF8') as f:
        f.write(json.dumps(result_dict, indent=2, sort_keys=False))

    print('ReID Done')
