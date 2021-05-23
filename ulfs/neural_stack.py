import torch
import numpy as np


class NeuralStack(object):
    """
    we're going to bound the stack with max_steps steps

    since we train RNNs with BTT, this seems not too crazy, for now, perhaps?

    since we want to do batches, this is going to work with batches

    the batch size needs to be fixed at the first time step

    well, it's fixed when one does 'reset'

    unlike pytorch/cudnn rnns, we're going to use [batch_size][seq_len][embedding_size]
    dimensions order
    """
    def __init__(self, embedding_size, max_steps):
        self.embedding_size = embedding_size
        self.max_steps = max_steps
        self.enable_cuda = False
        self.torch_constr = torch

    def cuda(self):
        self.enable_cuda = True
        self.torch_constr = torch.cuda
        return self

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.empty = self.torch_constr.Tensor(batch_size, self.embedding_size).zero_()
        self.V = self.torch_constr.Tensor(batch_size, self.max_steps, self.embedding_size).zero_()
        self.S = self.torch_constr.Tensor(batch_size, self.max_steps).zero_()
        self.T = 0

    def push(self, s, v):
        """
        we'll push 's' amount of 'v' onto the stack
        note that s and v are batch_size tensors
        """
        assert self.T + 1 < self.max_steps  # should we do something less catastrophic?
        self.V[np.arange(self.batch_size), self.T] = v
        self.S[np.arange(self.batch_size), self.T] = s
        self.T += 1

    # def read(self, s):
    #     r = self.torch_constr.Tensor(self.batch_size, self.max_steps).zero_()
        # assume all always alive for now...

    def pop(self, s):
        """
        so, what we need to do is to take the incoming strength values (batch_size long),
        and work down the strength values (per timestep, and per batch item), to get a read
        vector
        then we're going to read that value
        as go down the strength vector, we're also going to decrease it by the amount we're reading
        """
        assert(s.min() > -1e-8)
        assert(s.max() <= 1 + 1e-8)
        res = self.torch_constr.Tensor(self.batch_size, self.embedding_size).zero_()
        s = s.clone()
        for t in range(self.T - 1, -1, -1):
            ge_S_idxes = (s >= self.S[:, t]).view(-1).nonzero().long().view(-1)
            lt_S_idxes = (s < self.S[:, t]).view(-1).nonzero().long().view(-1)
            if len(ge_S_idxes) > 0:
                s[ge_S_idxes] = s[ge_S_idxes] - self.S[ge_S_idxes, t]
                value_to_add = self.S[ge_S_idxes, t].view(-1, 1).expand(-1, self.embedding_size) * self.V[ge_S_idxes, t]
                res[ge_S_idxes] = res[ge_S_idxes] + value_to_add
                self.S[ge_S_idxes, t] = 0
            if len(lt_S_idxes) > 0:
                value_to_add = s[lt_S_idxes].view(-1, 1).expand(-1, self.embedding_size) * self.V[lt_S_idxes, t]
                self.S[lt_S_idxes, t] = self.S[lt_S_idxes, t] - s[lt_S_idxes]
                s[lt_S_idxes] = 0
                res[lt_S_idxes] = res[lt_S_idxes] + value_to_add
        return res

    def read(self, s):
        """
        like pop, but without erasing S
        """
        assert(s.min() > -1e-8)
        assert(s.max() <= 1 + 1e-8)
        res = self.torch_constr.Tensor(self.batch_size, self.embedding_size).zero_()
        s = s.clone()
        for t in range(self.T - 1, -1, -1):
            ge_S_idxes = (s >= self.S[:, t]).view(-1).nonzero().long().view(-1)
            lt_S_idxes = (s < self.S[:, t]).view(-1).nonzero().long().view(-1)
            if len(ge_S_idxes) > 0:
                s[ge_S_idxes] = s[ge_S_idxes] - self.S[ge_S_idxes, t]
                value_to_add = self.S[ge_S_idxes, t].view(-1, 1).expand(-1, self.embedding_size) * self.V[ge_S_idxes, t]
                res[ge_S_idxes] = res[ge_S_idxes] + value_to_add
            if len(lt_S_idxes) > 0:
                value_to_add = s[lt_S_idxes].view(-1, 1).expand(-1, self.embedding_size) * self.V[lt_S_idxes, t]
                s[lt_S_idxes] = 0
                res[lt_S_idxes] = res[lt_S_idxes] + value_to_add
        return res

    def sieve(self, alive_idxes):
        """
        move the stack etc into smaller tensors, containing only the values from the
        batch indexes given in alive_idxes

        alive_idxes should be 1 dimensional (as per the assert)

        for now, we explicitly disallow empty alive_idxes, ie length 0, or 0 dimensional
        """
        assert len(alive_idxes.size()) == 1
        self.batch_size = alive_idxes.size()[0]
        self.V = self.V[alive_idxes]
        self.S = self.S[alive_idxes]
