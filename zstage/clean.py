import os
import shutil
import time

root = '/nfs3-p2/zsxm/naic/rematch/test_huffman'
time_str = time.strftime("%m-%d_%H:%M:%S", time.localtime())

del_folder_list = []

if os.path.exists(os.path.join(root, './compressed_query_feature')):
    os.rename(os.path.join(root, './compressed_query_feature'), os.path.join(root, f'./compressed_query_feature+{time_str}'))
    del_folder_list.append(os.path.join(root, f'./compressed_query_feature+{time_str}'))

if os.path.exists(os.path.join(root, './reconstructed_query_feature')):
    os.rename(os.path.join(root, './reconstructed_query_feature'), os.path.join(root, f'./reconstructed_query_feature+{time_str}'))
    del_folder_list.append(os.path.join(root, f'./reconstructed_query_feature+{time_str}'))

if os.path.exists(os.path.join(root, './reid_results')):
    os.rename(os.path.join(root, './reid_results'), os.path.join(root, f'./reid_results+{time_str}'))
    del_folder_list.append(os.path.join(root, f'./reid_results+{time_str}'))

print('Rename Folders Done!')

for folder in del_folder_list:
    shutil.rmtree(folder)

print('Delete Folders Done!')