import numpy as np
import torch


class DistractorsBatchReader(object):
    """
    draws batches from the (shuffled) dataset, and also distractors batches

    The distractors batches will be `num_distractors` longer than the examples batches
    """

    def __init__(self, data_tensor, batch_size, num_distractors):
        self.data_tensor = data_tensor
        self.N = self.data_tensor.size()[0]
        self.batch_size = batch_size
        self.num_batches = self.N // batch_size
        self.num_distractors = num_distractors
        self.shuffle_idxes1 = np.random.choice(self.N, self.N, replace=False)
        self.shuffled_dataset = self.data_tensor[self.shuffle_idxes1]
        self.enable_cuda = False

    def cuda(self):
        self.enable_cuda = True
        print('cuda enabled for distractors batch reader')

    def __iter__(self):
        for b in range(self.num_batches):
            distractor_pool_mask = torch.ones(self.N)
            distractor_pool_mask[b * self.batch_size: (b + 1) * self.batch_size] = 0
            distractor_pool_idxes = distractor_pool_mask.nonzero().view(-1)
            distractor_pool_sub_idxes = np.random.choice(
                self.N - self.batch_size, self.num_distractors * self.batch_size, replace=False)
            distractor_idxes = distractor_pool_idxes[distractor_pool_sub_idxes]

            examples_batch = self.shuffled_dataset[b * self.batch_size: (b + 1) * self.batch_size]
            distractors = self.shuffled_dataset[distractor_idxes].view(self.batch_size, self.num_distractors, -1)
            merged = torch.FloatTensor(self.batch_size, 1 + self.num_distractors, *list(self.data_tensor[0].size()))
            merged[:, 0] = examples_batch
            merged[:, 1:] = distractors
            if self.enable_cuda:
                examples_batch = examples_batch.cuda()
                distractors = distractors.cuda()
                merged = merged.cuda()
            batch = {
                'examples': examples_batch,
                'distractors': distractors,
                'merged': merged
            }
            yield batch


class DisjointDistractorsBatchReader(object):
    """
    Assume that examples tensor and distractors tensor are entirely disjoint

    Assumes distractors not enough, so samples from them *with replacement*
    """
    def __init__(self, examples_data, distractors_data, batch_size, num_distractors):
        self.examples_data = examples_data
        self.distractors_data = distractors_data
        self.N_examples = self.examples_data.size()[0]
        self.N_distractors = self.distractors_data.size()[0]
        self.batch_size = batch_size
        self.num_batches = self.N_examples // batch_size
        self.num_distractors = num_distractors

        self.shuffled_examples = self._shuffle(self.examples_data)
        self.enable_cuda = False

    def cuda(self):
        self.enable_cuda = True
        print('cuda enabled for distractors batch reader')

    def _shuffle(self, tensor):
        N = tensor.size()[0]
        idxes = np.random.choice(N, N, replace=False)
        shuffled = tensor[idxes]
        return shuffled

    def __iter__(self):
        for b in range(self.num_batches):
            examples_batch = self.shuffled_examples[b * self.batch_size: (b + 1) * self.batch_size]

            # sample with replacement ... (since there arent enough)
            distractor_idxes = np.random.choice(
                self.N_distractors, self.num_distractors * self.batch_size, replace=True)
            distractors = self.distractors_data[distractor_idxes]
            distractors = distractors.view(self.batch_size, self.num_distractors, -1)
            merged = torch.FloatTensor(self.batch_size, 1 + self.num_distractors, *list(self.examples_data[0].size()))
            merged[:, 0] = examples_batch
            merged[:, 1:] = distractors
            if self.enable_cuda:
                examples_batch = examples_batch.cuda()
                distractors = distractors.cuda()
                merged = merged.cuda()
            batch = {
                'examples': examples_batch,
                'distractors': distractors,
                'merged': merged
            }
            yield batch
