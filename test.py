# from datasets.preliminary_dataset import PreliminaryDataset, PreliminaryBatchSampler
# import random

# dataset = PreliminaryDataset('/nfs3-p1/zsxm/naic/preliminary/train')
# batchsampler = PreliminaryBatchSampler(dataset, 142)

# random.seed(2)
# test_count = 0
# while True:
#     batchs = []
#     t_batch = []
#     for i, b in enumerate(batchsampler):
#         print(i, b)
#         batchs.append(b)
#         t_batch.extend(b)
#         b_len = 0
#         for idx in b:
#             b_len += dataset.idx2len[idx]
#         assert b_len != 0 and b_len <= batchsampler.batch_size, str(b_len)+str(b)
#     assert len(t_batch) == 15000
#     test_count +=1
#     print(test_count)



# import os
# import numpy as np
# from collections import OrderedDict
# import json
# from tqdm import tqdm
# from scipy.spatial.distance import cdist

# query_feature_A_dir = '/nfs3-p2/zsxm/naic/preliminary/test_A/query_feature_A'
# gallery_feature_A_dir = '/nfs3-p2/zsxm/naic/preliminary/test_A/gallery_feature_A'
# query_reshape_A = np.load('/nfs3-p2/zsxm/naic/preliminary/test_A/query_reshape_A.npy')
# gallery_reshape_A = np.load('/nfs3-p2/zsxm/naic/preliminary/test_A/gallery_reshape_A.npy')
# from re_ranking.re_ranking_my import calc_dist
# calc_dist(query_reshape_A, gallery_reshape_A, 100, 1000, True)


# import os
# import numpy as np
# from collections import OrderedDict
# import json
# from tqdm import tqdm
# from scipy.spatial.distance import cdist

# query_reshape_A = np.load('/nfs3-p2/zsxm/naic/preliminary/test_A/query_reshape_A.npy')
# gallery_reshape_A = np.load('/nfs3-p2/zsxm/naic/preliminary/test_A/gallery_reshape_A.npy')
# from re_ranking.re_ranking_my import re_ranking
# re_ranking(query_reshape_A, gallery_reshape_A, 100, 30, 0.3, 1000)


# import os
# import numpy as np
# import tables
# dis_path = '/nfs3-p2/zsxm/naic/preliminary/test_A/dis'
# original_dist_path = os.path.join(dis_path, 'original_dist.hdf5')
# original_dist_file = tables.open_file(original_dist_path, mode='w')
# earray = original_dist_file.create_earray()
# tables.open_file()
# filters = tables.Filters(complevel=5, complib='blosc')
# original_dist_file.create_earray()
# original_dist_file.create_carray()
# original_dist_file.


import os
import numpy as np
from collections import OrderedDict
import json
from tqdm import tqdm
from scipy.spatial.distance import cdist
query_feature_A_dir = '/nfs3-p2/zsxm/naic/preliminary/test_A/query_feature_A'
gallery_feature_A_dir = '/nfs3-p2/zsxm/naic/preliminary/test_A/gallery_feature_A'
query_reshape_A = np.load('/nfs3-p2/zsxm/naic/preliminary/test_A/query_reshape_A.npy')
gallery_reshape_A = np.load('/nfs3-p2/zsxm/naic/preliminary/test_A/gallery_reshape_A.npy')
from re_ranking.re_ranking_pytable import re_ranking
res = re_ranking(20000, 428794, 100, 30, 0.3, 1000)
np.save('/nfs3-p2/zsxm/naic/preliminary/test_A/rerank_res.npy', res)