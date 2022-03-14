import os
import json
import gc

import numpy as np
import torch

def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


def reid(bytes_rate):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    gallery_fea_dir = 'gallery_feature'
    reid_results_path = 'reid_results/{}.json'.format(bytes_rate)
    os.makedirs(os.path.dirname(reid_results_path), exist_ok=True)

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

    cos = torch.nn.CosineSimilarity(dim=1)

    # dists = np.linalg.norm(reconstructed_query_fea_all - gallery_fea_all, ord=2, axis=2)[:, :top_num]
    # indexes = np.argsort(dists, axis=1)

    result_dict = {}
    gallery_names_array = np.array(list(map(lambda _: _.rsplit('.', 1)[0] + '.png', gallery_names)))
    del gallery_names
    for query_idx, query_name in enumerate(query_names):
        query_name = query_name.rsplit('.', 1)[0] + '.png'
        try:
            res = cos(reconstructed_query_fea_all[query_idx].unsqueeze(0), gallery_fea_all)
        except RuntimeError:
            reconstructed_query_fea_all = reconstructed_query_fea_all.cpu()
            gallery_fea_all = gallery_fea_all.cpu()
            res = cos(reconstructed_query_fea_all[query_idx].unsqueeze(0), gallery_fea_all)
        indexes = torch.argsort(res, descending=True)[:top_num].cpu().numpy()

        result_dict[query_name] = gallery_names_array[indexes].tolist()
        gc.collect()

    with open(reid_results_path, 'w', encoding='UTF8') as f:
        f.write(json.dumps(result_dict, indent=2, sort_keys=False))

    print('ReID Done')
