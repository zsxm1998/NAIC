import os
import copy
import random
from os.path import join
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

class PreliminaryDataset(Dataset):
    def __init__(self, dir):
        self.file_dir = join(dir, 'train_feature')
        self.datas = {}
        with open(join(dir, 'train_list.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.strip()
                elems = line.split(' ')
                self.datas.setdefault(int(elems[1]), []).append(elems[0])
        self.len_idx_sets, self.idx2len = {}, {}
        self.q_len, self.max_len = 0, -1
        for k, v in self.datas.items():
            l = len(v)//2
            self.len_idx_sets.setdefault(l, set()).add(k)
            self.idx2len[k] = l
            self.q_len += l
            self.max_len = max(self.max_len, l)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        q_file_list = random.sample(self.datas[index], len(self.datas[index])//2)
        k_file_list = [f for f in self.datas[index] if f not in q_file_list]
        q_list, k_list = [], []
        for q_file in q_file_list:
            q_list.append(torch.from_numpy(np.fromfile(join(self.file_dir, q_file), dtype='<f4')))
        for k_file in k_file_list:
            k_list.append(torch.from_numpy(np.fromfile(join(self.file_dir, k_file), dtype='<f4')))
        return q_list, k_list, torch.tensor(index, dtype=torch.long)


class PreliminaryBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        assert batch_size >= dataset.max_len, f'Batch Size should not less than dataset.max_len({dataset.max_len}), but is {batch_size}'

    def __iter__(self):
        min_batch_num = self.dataset.q_len // self.batch_size
        batch_num = 0
        index_list = list(range(len(self.dataset)))
        random.shuffle(index_list)
        len_idx_sets = copy.deepcopy(self.dataset.len_idx_sets)
        len_list = sorted(list(len_idx_sets.keys()), reverse=True)
        next_dict = {}
        for i in range(len(len_list)-1):
            next_dict[len_list[i]] = len_list[i+1]
        next_dict[len_list[-1]] = -1
        batch = []
        batch_len = 0
        for index in index_list:
            idx_len = self.dataset.idx2len[index]
            if batch_len + idx_len <= self.batch_size:
                batch.append(index)
                len_idx_sets[idx_len].remove(index)
                batch_len += idx_len
                if batch_len == self.batch_size:
                    batch_num += 1
                    thresh = batch_num / min_batch_num
                    while random.random() < thresh and len(len_idx_sets[0]) != 0:
                        select_index = random.choice(list(len_idx_sets[0]))
                        batch.append(select_index)
                        len_idx_sets[0].remove(select_index)
                        index_list.remove(select_index)
                    yield batch
                    batch = []
                    batch_len = 0
            elif batch_len + idx_len > self.batch_size:
                select_len = self.batch_size - batch_len
                while select_len not in next_dict.keys():
                    select_len -= 1
                empty_list = []
                while select_len != -1 and len(len_idx_sets[select_len]) == 0:
                    empty_list.append(select_len)
                    select_len = next_dict[select_len]
                if len(empty_list) != 0:
                    empty_list.pop()
                for el in empty_list:
                    next_dict[el] = select_len
                batch_num += 1
                if select_len != -1:
                    if select_len != 0:
                        select_index = random.choice(list(len_idx_sets[select_len]))
                        batch.append(select_index)
                        len_idx_sets[select_len].remove(select_index)
                        index_list.remove(select_index)
                    else:
                        thresh = batch_num / min_batch_num
                        while random.random() < thresh and len(len_idx_sets[select_len]) != 0:
                            select_index = random.choice(list(len_idx_sets[select_len]))
                            batch.append(select_index)
                            len_idx_sets[select_len].remove(select_index)
                            index_list.remove(select_index)
                yield batch
                batch = [index]
                len_idx_sets[idx_len].remove(index)
                batch_len = idx_len
        if len(batch) != 0:
            yield batch


def preliminary_collate_fn(input_list): #[(q_list, k_list, label), (q_list, k_list, label)...]
    q