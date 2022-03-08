import os
import glob

import numpy as np


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


# def compress_feature(fea: np.ndarray, target_bytes: int, path: str):
#     assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
#     fea.astype('<f4')[: target_bytes // 4].tofile(path)
#     return True
def compress_feature(fea: np.ndarray, target_bytes: int, path: str):
    fea.astype('<f4').tofile(path)
    return True


def compress(bytes_rate):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = 'query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))
    assert(len(query_fea_paths) != 0)
    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path)
        fea = read_feature_file(query_fea_path)
        compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
        compress_feature(fea, bytes_rate, compressed_fea_path)

    print('Compression Done')
