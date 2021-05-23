"""
Handle alive/dead masks
"""
import torch


class AliveSieve(object):
    def __init__(self, batch_size, enable_cuda=False, global_idxes=None):
        """
        assume all alive to start with, with given batch_size
        """
        self.batch_size = batch_size
        self.enable_cuda = enable_cuda
        self.type_constr = torch.cuda if enable_cuda else torch
        self.alive_mask = self.type_constr.ByteTensor(batch_size).fill_(1)
        if enable_cuda:
            self.alive_mask = self.alive_mask.cuda()
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)
        """
        global_idxes are the indexes of the current members of alivesieve, in the original batch
        basically, we popualte it initially with 0...batch_size-1 , and then
        sieve this too each time we call sieve_self
        """
        if global_idxes is None:
            global_idxes = self.alive_idxes.clone()
        self.global_idxes = global_idxes

    def cuda(self):
        self.enable_cuda = True
        return self

    @staticmethod
    def mask_to_idxes(mask):
        return mask.view(-1).nonzero(as_tuple=False).long().view(-1)

    def mark_dead(self, dead_mask):
        """
        sets the mask to 0 at relevant positions
        doesnt remove them yet, ie doesnt call 'sieve'
        """
        # if len(dead_mask) == 0:
        #     return
        # print('dead_mask', dead_mask)
        dead_idxes = self.mask_to_idxes(dead_mask.view(-1))
        # print('dead_idxes', dead_idxes)
        if len(dead_idxes) == 0:
            return
        self.alive_mask[dead_idxes] = 0
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)

    def get_dead_idxes(self):
        dead_mask = 1 - self.alive_mask
        if len(dead_mask) == 0:
            return []
        return dead_mask.nonzero().long().view(-1)

    def get_alive_idxes(self):
        if len(self.alive_mask) == 0:
            return []
        return self.alive_mask.nonzero().long().view(-1)

    def any_alive(self):
        return self.alive_mask.max() == 1

    def all_dead(self):
        return self.alive_mask.max() == 0

    def set_global_dead(self, target, v):
        """
        What this does is assign v to target, masked to only modify items marked as dead
        It assumes that target is a global tensor, ie hasnt been sieved
        So we're going to use global_idxes to index into this
        """
        dead_idxes = self.get_dead_idxes()
        if len(dead_idxes) == 0:
            return
        target[self.global_idxes[dead_idxes]] = v

    def self_sieve_(self):
        """
        removes all dead from alive_mask
        basically equivalent to resetting the batch_size, and
        recreating an all-1s mask
        """
        self.global_idxes = self.global_idxes[self.alive_idxes]

        self.batch_size = self.alive_mask.int().sum().item()
        self.alive_mask = self.type_constr.ByteTensor(self.batch_size).fill_(1)
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)

    def sieve_tensor(self, t):
        """
        returns t sieved by current alive mask (ie anything not alive is
        not returned)
        note: the returned tensor is NOT guaranteed to be a reference into
        the old one. In fact, I'm fairly sure it will never in fact be (but
        this isnt guaranteed either; I havent checked...)
        """
        return t[self.alive_idxes]

    def sieve_list(self, alist):
        """
        returns alist sieved by current alive mask (ie anything not alive is
        not returned)
        """
        return [alist[b] for b in self.alive_idxes]


class SievePlayback(object):
    """
    Given a list of alive_masks, will loop through these, providing a tuple of
    t, and the global idxes at each loop step
    """
    def __init__(self, alive_masks, enable_cuda):
        self.alive_masks = alive_masks
        self.type_constr = torch.cuda if enable_cuda else torch

    def __iter__(self):
        batch_size = self.alive_masks[0].size()[0]
        global_idxes = self.type_constr.ByteTensor(batch_size).fill_(1).nonzero().long().view(-1)
        T = len(self.alive_masks)
        for t in range(T):
            self.batch_size = len(global_idxes)
            self.alive_mask = self.alive_masks[t]
            self.alive_idxes = self.alive_mask.nonzero().long().view(-1)
            self.all_dead = self.alive_mask.max() == 0
            yield t, global_idxes
            mask = self.alive_masks[t]
            if mask.max() == 0:
                return
            global_idxes = global_idxes[mask.nonzero().long().view(-1)]
