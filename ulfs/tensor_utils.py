import string
from typing import Iterable, Dict

import torch
import torch.nn.functional as F
import numpy as np


class DimMerger(object):
    def __init__(self):
        self.size1, self.size2 = None, None

    def merge(self, target: torch.Tensor, dim1: int, dim2: int) -> torch.Tensor:
        assert dim2 == dim1 + 1
        if self.size1 is None:
            self.size1 = target.size(dim1)
            self.size2 = target.size(dim2)
        else:
            assert self.size1 == target.size(dim1)
            assert self.size2 == target.size(dim2)
        return merge_dims(target, dim1, dim2)

    def resplit(self, target: torch.Tensor, dim: int) -> torch.Tensor:
        assert target.size(dim) == self.size1 * self.size2
        return split_dim(target, dim, self.size1, self.size2)


def split_dim(tensor, target_dim, size1, size2):
    """
    splits dimensionn target_dim of tensor tensor into two dims [size1][size2]
    """
    d = list(tensor.size())
    assert d[target_dim] == size1 * size2
    d_new = d[:target_dim] + [size1, size2] + d[target_dim + 1:]
    return tensor.view(*d_new)


def merge_dims(tensor, d1, d2):
    """
    merges dimensions d1 and d1 + 1 of tensor
    """
    D = len(tensor.size())
    if d1 < 0:
        d1 = D + d1
    if d2 < 0:
        d2 = D + d2
    assert d2 == d1 + 1
    d = list(tensor.size())
    d_new = d[:d1] + [d[d1] * d[d1 + 1]] + d[d1 + 2:]
    return tensor.view(*d_new)


def softmax(inputs, tau):
    inputs = inputs / tau
    inputs = inputs - inputs.max(dim=-1)[0].unsqueeze(1)
    r = inputs.exp() / inputs.exp().sum(dim=-1).unsqueeze(1)
    return r


def softmax_spatial_planar(images):
    """
    normalizes over each plane, over spatial locations
    """
    images_max = images.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1)
    images = images - images_max
    images_exp = images.exp()
    images_exp_sum = images_exp.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    res = images_exp / images_exp_sum
    return res


def invert_s2o(s2o):
    """
    given you've done:

        sorted, s2o = sometensor.sort()

    ... so that s2o is the index in the original tensor of position s in the sorted tensor,
    then you can call:

        o2s = invert_s2o(s2o)

    ... to get o2s, which is the index in the sorted tensor of index o in the original tensor

    assumes that s and o are 1-dimensional
    """
    assert len(s2o.size()) == 1
    N = s2o.size(0)
    sequential = torch.ones(N, device=s2o.device, dtype=torch.int64).cumsum(dim=-1) - 1
    o2s = torch.zeros_like(sequential)
    o2s.scatter_(0, s2o, sequential)
    return o2s


def pack_sequences(seqs, lens):
    """
    doesnt need sequences to be sorted by length. handles this for us/you

    sequences should be [M][N][E]

    returns the inverse sequence mapping to unpack later
    """
    lens = lens.view(-1)
    N = lens.size(0)
    lens_sorted, lens_sorted_idxes = lens.sort(descending=True)
    # notation: s is sorted index; i is incoming index
    # s2i maps from s => i
    # i2s maps from i => s
    sequential = torch.ones(N, device=lens.device, dtype=torch.int64).cumsum(dim=-1) - 1
    s2i = lens_sorted_idxes
    i2s = torch.zeros_like(sequential)
    i2s.scatter_(0, lens_sorted_idxes, sequential)
    seqs_sorted = seqs.index_select(dim=1, index=s2i)
    packed = torch.nn.utils.rnn.pack_padded_sequence(seqs_sorted, lens_sorted)
    reverse_idxes = i2s
    return packed, reverse_idxes


def unpack_sequences(packed, reverse_idxes, total_length):
    """
    unpacked seqs assumed to be [N][N][E]
    """
    seqs, lens = torch.nn.utils.rnn.pad_packed_sequence(packed, total_length=total_length)
    seqs = seqs.index_select(dim=1, index=reverse_idxes)
    seq_lens = lens[reverse_idxes]
    return seqs, seq_lens


def l2_normalize(x, dim=-1, eps=1e-6):
    xdiv = torch.norm(x, p=2, dim=dim, keepdim=True)
    xdiv = F.threshold(xdiv, eps, eps)
    x = x / xdiv
    return x


def masked_get(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    returns a tensor with only the values from input where
    mask is 1
    (safer than just multiplying, because avoids nans etc)
    """
    assert mask.dtype == torch.bool
    res = input.clone()
    res.masked_fill_(~mask, 0)
    return res


def tensor_to_str(t, vocab=' ' + string.ascii_lowercase):
    """
    t is a tensor of Longs. We assume that 0 is space, and the next
    26 values are letters
    we convert t into a string
    """
    res = ''
    if len(t.size()) != 1:
        print('t should be one-dim, but has size', t.size())
    assert len(t.size()) == 1
    N = t.size(0)
    for i in range(N):
        res += vocab[t[i].item()]
    return res


def tensor_to_2dstr(t, vocab=' ' + string.ascii_lowercase):
    assert len(t.size()) == 2
    N = t.size(0)
    res = ''
    for n in range(N):
        res += tensor_to_str(t[n], vocab=vocab) + '\n'
    return res


def multi_onehot(sizes_l, tensor):
    """
    for a multi LongTensor representing a mini-batch, we 'fluff up' the tensor
    into one-hot, based on the sizes in sizes_l
    """
    batch_size = tensor.size()[0]
    torch_constr = torch.cuda if tensor.is_cuda else torch
    res = torch_constr.FloatTensor(batch_size, np.sum(sizes_l)).zero_()
    sub = res
    for i, size in enumerate(sizes_l):
        sub[np.arange(batch_size), tensor[:, i]] = 1
        sub = sub[:, size:]
    return res


def multi_argmax(sizes_l, tensor):
    """
    for a 2d tensor representing a mini-batch, we extract an argmax for each set of columns
    delineated by sizes_l

    For example, if sizes_l is [2, 3], then the result will have 2 columns, one is argmax
    over the first two columns; and the next column is argmax over the next three columns
    """
    torch_constr = torch.cuda if tensor.is_cuda else torch
    batch_size = tensor.size()[0]
    ret_K = len(sizes_l)
    res = torch_constr.LongTensor(batch_size, ret_K).zero_()
    for k, size in enumerate(sizes_l):
        _, this_argmax = tensor[:, :size].max(dim=-1)
        tensor = tensor[:, size:]
        res[:, k] = this_argmax
    return res


def concat(one, two):
    return torch.cat([one, two], dim=-1)


def Hadamard(one, two):
    if one.size() != two.size():
        raise Exception('size mismatch %s vs %s' % (str(list(one.size())), str(list(two.size()))))
    res = one * two
    assert res.numel() == one.numel()
    return res


def view_by_tensor(target, pos):
    view = target
    for d, i in enumerate(pos.tolist()):
        view = view.narrow(d, i, 1)
    return view


def lengths_to_mask(lengths, max_len):
    """
    This mask has a 1 for every location where we should calculate a loss

    lengths include the null terminator. for example the following is length 1:

    0 0 0 0 0 0

    The following are each length 2:

    1 0 0 0 0
    3 0 0 0 0

    if max_len is 3, then the tensor will be 3 wide. The longest tensors will look
    like:

    1 2 0
    3 4 0
    5 7 0

    Whils the rnn might not generate the final 0 each time, this is an error

    The mask for these length 3 utterances will be all 1s:

    1 1 1
    1 1 1

    """
    assert len(lengths.size()) == 1
    N = lengths.size()[0]
    cumsum = torch.zeros(N, max_len, device=lengths.device, dtype=torch.int64).fill_(1)
    cumsum = cumsum.cumsum(dim=-1)
    l_expand = lengths.view(N, 1).expand(N, max_len)
    in_alive_mask = l_expand > cumsum - 1
    return in_alive_mask


def add_terminator_zeros(tokens, lengths):
    """
    see description of lengths_to_mask
    basically, this will set the token to 0 just before the first 0 of each
    example's mask

    eg, if we have length 2, and incoming tokens looks like:

    3 5 2 3

    This function will set 0 at position 1:

    3 0 2 3

    the mask from lengths_to_mask will look like:

    1 1 0 0

    (ie includes the null terminator 0 in the 1s part)
    """
    N = tokens.size()[0]
    tokens[np.arange(N), lengths - 1] = 0


def params_means_abs_diff(params1: Iterable[torch.Tensor], params2: Iterable[torch.Tensor]) -> float:
    """
    params1 and params2 should both be lists of tensors
    we will return the average of the absolute per-element difference
    between the params
    """
    numels = 0
    abs_sum = 0.0
    for p1, p2 in zip(params1, params2):
        assert p1.numel() == p2.numel()
        numels += p1.numel()
        abs_sum += (p2 - p1).abs().sum().item()
    return abs_sum / numels


def named_params_means_abs_diff(
        params1: Dict[str, torch.Tensor], params2: Dict[str, torch.Tensor]) -> float:
    """
    params1 and params2 should both be lists of tensors
    we will return the average of the absolute per-element difference
    between the params
    """
    numels = 0
    abs_sum = 0.0
    assert len(params1) == len(params2)
    for k, p1 in params1.items():
        p2 = params2[k]
        assert p1.numel() == p2.numel()
        numels += p1.numel()
        abs_sum += (p2 - p1).abs().sum().item()
    return abs_sum / numels


def print_samples(target: torch.Tensor, num_samples: int = 20, prob: float = 0.01) -> None:
    if torch.rand(1).item() > prob:
        return
    tgt_flat = target.contiguous().view(-1)
    N = tgt_flat.size(0)
    sample_idxes = torch.from_numpy(np.random.choice(N, num_samples, replace=True))
    samples = tgt_flat[sample_idxes]
    print(['%.3f' % v for v in samples.tolist()])


def idxes_to_onehot(idxes: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Given incoming tensor with dimensions [*] and vocab_size vocab_size,
    adds additional final dimension of size vocab_size, and converts each
    index into one-hot over this dimension
    """
    onehot_size = list(idxes.size()) + [vocab_size]
    onehot = torch.zeros(onehot_size, device=idxes.device, dtype=torch.float32)
    onehot.scatter_(dim=-1, index=idxes.unsqueeze(-1), value=1.0)
    return onehot


def make_hard(target: torch.Tensor, verify=False):
    """
    returns hard version of target, with gradient stripped
    assumes we apply to last dim

    if verify, then asssert that really hard and no grad
    """
    _, toks = target.max(dim=-1)
    vocab_size = target.size(-1)
    hard = idxes_to_onehot(toks, vocab_size=vocab_size)
    assert list(hard.size()) == list(target.size())
    # check only 1s and 0s, ie hard
    if verify:
        assert (hard == 0).int().sum() + (hard == 1).int().sum() == hard.numel()
        assert not hard.requires_grad
    return hard
