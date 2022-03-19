import os
import json
import gc

import numpy as np
import torch
import torch.nn.functional as F

DIM_NUM = 1024
WEIGHT = [0.02, 0.56, 0.24, 0.1, 0.02, 0.01, 0.02, 0.02, 0.02]
# WEIGHT = [0.3, 0.6, 0.1]

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

def reid(bytes_rate, root='', method='rerank_pearson'):
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
    top_num = min(100, gallery_num)
    del reconstructed_query_fea_list, gallery_fea_list
    gc.collect()

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
    batch_size = 50 if 'l2' in method else 512
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
