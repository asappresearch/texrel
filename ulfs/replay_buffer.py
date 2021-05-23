import torch


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.enable_cuda = False

    def cuda(self):
        """
        note that the buffer itself is *always* in main memory, not on gpu
        but activating cuda means that tensors pulled out of the replay buffer
        are already cudarized
        """
        self.enable_cuda = True
        print('cuda enabled on replay buffer')
        return self

    def append(self, experience):
        """
        incoming experience should be cpu, not cuda
        """
        cpu_experience = {}
        for k, v in experience.items():
            if v is not None:
                v = v.detach().cpu()
            cpu_experience[k] = v
        self.buffer.append(cpu_experience)

    def shrink(self):
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def get_tensor(self, idxes, name, dtype=None):
        batch_size = idxes.numel()
        sample = self.buffer[0][name]
        if dtype is None:
            dtype = sample.dtype
        size = sample.size()
        res = torch.empty(size=[batch_size] + list(size), dtype=dtype)
        for j in range(batch_size):
            idx = idxes[j]
            experience = self.buffer[idx]
            res[j] = experience[name]
        if self.enable_cuda:
            res = res.cuda()
        return res

    def get_float_tensors(self, idxes, names):
        tensors = []
        for name in names:
            t = self.get_tensor(idxes, name, dtype=torch.float32)
            tensors.append(t)
        return tensors

    def get_float_tensor(self, idxes, name):
        return self.get_tensor(idxes, name, dtype=torch.float32)

    def get_int_tensor(self, idxes, name):
        return self.get_tensor(idxes, name, dtype=torch.int32)

    def __len__(self):
        return len(self.buffer)

    def get_float_tensor_and_mask(self, idxes, name, tensor_size):
        batch_size = idxes.numel()
        res = torch.zeros([batch_size] + list(tensor_size), dtype=torch.float32)
        mask = torch.ByteTensor(batch_size).zero_()
        for j in range(batch_size):
            idx = idxes[j]
            experience = self.buffer[idx]
            if experience[name] is not None:
                res[j] = experience[name]
                mask[j] = 1
        if self.enable_cuda:
            res = res.cuda()
            mask = mask.cuda()
        return res, mask
