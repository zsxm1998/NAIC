from datasets.preliminary_dataset import PreliminaryDataset, PreliminaryBatchSampler
import random

dataset = PreliminaryDataset('/nfs3-p1/zsxm/naic/preliminary/train')
batchsampler = PreliminaryBatchSampler(dataset, 142)

random.seed(2)
test_count = 0
while True:
    batchs = []
    t_batch = []
    for i, b in enumerate(batchsampler):
        print(i, b)
        batchs.append(b)
        t_batch.extend(b)
        b_len = 0
        for idx in b:
            b_len += dataset.idx2len[idx]
        assert b_len != 0 and b_len <= batchsampler.batch_size, str(b_len)+str(b)
    assert len(t_batch) == 15000
    test_count +=1
    print(test_count)