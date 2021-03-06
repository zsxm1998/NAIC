import os
import copy
import random
from os.path import join
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

class PreliminaryDataset(Dataset):
    def __init__(self, dir, memory=True):
        self.file_dir = join(dir, 'train_feature')
        self.memory = memory
        self.datas = {}
        if os.path.exists(join(dir, 'not_zero_dim.pt')):
            self.not_zero_dim = torch.load(join(dir, 'not_zero_dim.pt'))
        else:
            self.not_zero_dim = None
        with open(join(dir, 'train_list.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.strip()
                elems = line.split(' ')
                self.datas.setdefault(int(elems[1]), []).append(elems[0])
        self.instance_len = len(self.datas)
        self.len_idx_sets, self.idx2len = {}, {}
        self.q_len, self.max_len = 0, -1
        for k, v in self.datas.items():
            l = len(v)//2
            self.len_idx_sets.setdefault(l, set()).add(k)
            self.idx2len[k] = l
            self.q_len += l
            self.max_len = max(self.max_len, l)
        if memory:
            database_list = []
            self.file2idx = {}
            for i, data in enumerate(sorted(os.listdir(self.file_dir))):
                v = torch.from_numpy(np.fromfile(join(self.file_dir, data), dtype='<f4'))
                if self.not_zero_dim is not None:
                    v = v[self.not_zero_dim]
                database_list.append(v)
                self.file2idx[data] = i
            self.database = torch.stack(database_list)

    def __len__(self):
        return self.q_len

    def __getitem__(self, index):
        q_file_list = random.sample(self.datas[index], len(self.datas[index])//2)
        k_file_list = [f for f in self.datas[index] if f not in q_file_list]
        q_list, k_list = [], []
        for q_file in q_file_list:
            if self.memory:
                q_list.append(self.database[self.file2idx[q_file]])
            else:
                v = torch.from_numpy(np.fromfile(join(self.file_dir, q_file), dtype='<f4'))
                if self.not_zero_dim is not None:
                    v = v[self.not_zero_dim]
                q_list.append(v)
        for k_file in k_file_list:
            if self.memory:
                k_list.append(self.database[self.file2idx[k_file]])
            else:
                v = torch.from_numpy(np.fromfile(join(self.file_dir, k_file), dtype='<f4'))
                if self.not_zero_dim is not None:
                    v = v[self.not_zero_dim]
                k_list.append(v)
        return q_list, k_list, torch.tensor(index, dtype=torch.long)


class PreliminaryBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        assert batch_size >= dataset.max_len, f'Batch Size should not less than dataset.max_len({dataset.max_len}), but is {batch_size}'

    def __iter__(self):
        min_batch_num = self.dataset.q_len // self.batch_size
        batch_num = 0
        index_list = list(range(self.dataset.instance_len))
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
    q_label_list, k_label_list = [], []
    q_list, k_list = [], []
    for input in input_list:
        q_label, k_label = input[2].repeat(len(input[0])), input[2].repeat(len(input[1]))
        q_label_list.append(q_label)
        k_label_list.append(k_label)
        q_list.extend(input[0])
        k_list.extend(input[1])
    return torch.stack(q_list), torch.stack(k_list), torch.cat(q_label_list), torch.cat(k_label_list)