import os
import glob

import numpy as np


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    #assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


# def reconstruct_feature(path: str) -> np.ndarray:
#     fea = np.fromfile(path, dtype='<f4')
#     fea = np.concatenate(
#         [fea, np.zeros(2048 - fea.shape[0], dtype='<f4')], axis=0
#     )
#     return fea
def reconstruct_feature(path: str) -> np.ndarray:
    fea = np.fromfile(path, dtype='<f4')
    return fea


def reconstruct(bytes_rate):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))
    assert(len(compressed_query_fea_paths) != 0)
    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        reconstructed_fea = reconstruct_feature(compressed_query_fea_path)
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
        write_feature_file(reconstructed_fea, reconstructed_fea_path)

    print('Reconstruction Done')
