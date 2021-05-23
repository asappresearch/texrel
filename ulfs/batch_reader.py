import numpy as np


class BatchReader(object):
    def __init__(self, tensors_l, batch_size):
        # self.data_tensor = data_tensor
        self.tensors_l = tensors_l
        self.N = self.tensors_l[0].size()[0]
        self.batch_size = batch_size
        self.num_batches = self.N // batch_size
        self.shuffle_idxes1 = np.random.choice(self.N, self.N, replace=False)
        self.shuffled_tensors = []
        for t in self.tensors_l:
            self.shuffled_tensors.append(t[self.shuffle_idxes1])
        self.enable_cuda = False

    def cuda(self):
        self.enable_cuda = True
        print('cuda enabled for batch reader')

    def __iter__(self):
        for b in range(self.num_batches):
            batch = []
            for t in self.shuffled_tensors:
                t_batch = t[b * self.batch_size: (b + 1) * self.batch_size]
                if self.enable_cuda:
                    t_batch = t_batch.cuda()
                batch.append(t_batch)
            # examples_batch = self.shuffled_dataset[b * self.batch_size: (b + 1) * self.batch_size]
            # yield examples_batch
            yield batch
