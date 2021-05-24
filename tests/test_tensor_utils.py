import copy
from typing import Iterable, Dict

import torch
from torch import nn
import numpy as np
import pytest

from ulfs import tensor_utils


def test_split_dim():
    t = torch.Tensor(3, 5, 77, 5, 3)
    print('t.size()', t.size())
    t2 = tensor_utils.split_dim(t, 2, 7, 11)
    print('t2.size()')
    assert list(t2.size()) == [3, 5, 7, 11, 5, 3]

    assert t[2, 4, 0, 3, 2] == t2[2, 4, 0, 0, 3, 2]
    assert t[2, 4, 3, 3, 2] == t2[2, 4, 0, 3, 3, 2]
    assert t[2, 4, 11, 3, 2] == t2[2, 4, 1, 0, 3, 2]
    assert t[2, 4, 15, 3, 2] == t2[2, 4, 1, 4, 3, 2]
    assert t[2, 4, 39, 3, 2] == t2[2, 4, 3, 6, 3, 2]


def test_merge_dims():
    t = torch.Tensor(3, 5, 7, 11, 5, 3)
    print('t.size()', t.size())
    t2 = tensor_utils.merge_dims(t, 2, 3)
    print('t2.size()')
    assert list(t2.size()) == [3, 5, 77, 5, 3]

    assert t2[2, 4, 0, 3, 2] == t[2, 4, 0, 0, 3, 2]
    assert t2[2, 4, 3, 3, 2] == t[2, 4, 0, 3, 3, 2]
    assert t2[2, 4, 11, 3, 2] == t[2, 4, 1, 0, 3, 2]
    assert t2[2, 4, 15, 3, 2] == t[2, 4, 1, 4, 3, 2]
    assert t2[2, 4, 39, 3, 2] == t[2, 4, 3, 6, 3, 2]


def test_merge_dims_neg():
    t = torch.Tensor(3, 5, 7, 11, 5, 3)
    print('t.size()', t.size())
    t2 = tensor_utils.merge_dims(t, -4, -3)
    print('t2.size()')
    assert list(t2.size()) == [3, 5, 77, 5, 3]

    assert t2[2, 4, 0, 3, 2] == t[2, 4, 0, 0, 3, 2]
    assert t2[2, 4, 3, 3, 2] == t[2, 4, 0, 3, 3, 2]
    assert t2[2, 4, 11, 3, 2] == t[2, 4, 1, 0, 3, 2]
    assert t2[2, 4, 15, 3, 2] == t[2, 4, 1, 4, 3, 2]
    assert t2[2, 4, 39, 3, 2] == t[2, 4, 3, 6, 3, 2]


def test_dim_merger():
    a = torch.rand(3, 5, 7, 11)
    b = torch.rand(5, 7, 9)
    c_flat = torch.rand(3, 4, 5 * 7, 15)
    dim_merger = tensor_utils.DimMerger()
    a_flat = dim_merger.merge(a, 1, 2)
    b_flat = dim_merger.merge(b, 0, 1)
    c = dim_merger.resplit(c_flat, 2)
    assert list(a_flat.size()) == [3, 5 * 7, 11]
    assert list(b_flat.size()) == [5 * 7, 9]
    assert list(c.size()) == [3, 4, 5, 7, 15]


@pytest.mark.parametrize(
    "a,a_dim1,b,b_dim1,ok",
    [
        (torch.rand(3, 5, 2), 1, torch.rand(3, 5, 3), 1, False),
        (torch.rand(3, 5, 2), 1, torch.rand(3, 5, 2), 1, True),
        (torch.rand(3, 5, 2), 1, torch.rand(5, 2, 4), 0, True),
        (torch.rand(3, 5, 2), 1, torch.rand(5, 2, 4), 1, False),
        (torch.rand(3, 5, 2), 1, torch.rand(4, 2, 4), 0, False),
        (torch.rand(3, 5, 2), 1, torch.rand(5, 3, 4), 0, False),
    ]
)
def test_dim_merger2s(a, a_dim1, b, b_dim1, ok):
    dim_merger = tensor_utils.DimMerger()
    if ok:
        dim_merger.merge(a, a_dim1, a_dim1 + 1)
        dim_merger.merge(b, b_dim1, b_dim1 + 1)
    else:
        with pytest.raises(Exception):
            dim_merger.merge(a, a_dim1, a_dim1 + 1)
            dim_merger.merge(b, b_dim1, b_dim1 + 1)


@pytest.mark.parametrize(
    "a,a_dim1,b,b_dim1,ok",
    [
        (torch.rand(3, 5, 2), 1, torch.rand(3, 10), 1, True),
        (torch.rand(3, 5, 2), 1, torch.rand(3, 11), 1, False),
        (torch.rand(3, 5, 2), 1, torch.rand(3, 10), 0, False),
        (torch.rand(3, 5, 2), 1, torch.rand(10, 3), 1, False),
        (torch.rand(3, 5, 2), 1, torch.rand(10, 3), 0, True),
    ]
)
def test_dim_merger_merge(a, a_dim1, b, b_dim1, ok):
    dim_merger = tensor_utils.DimMerger()
    if ok:
        dim_merger.merge(a, a_dim1, a_dim1 + 1)
        dim_merger.resplit(b, b_dim1)
    else:
        with pytest.raises(Exception):
            dim_merger.merge(a, a_dim1, a_dim1 + 1)
            dim_merger.resplit(b, b_dim1)


def test_invert_s2o():
    lengths = [7, 5, 9, 13, 11]
    t = torch.LongTensor(lengths)
    print('t', t, t.dtype)

    s, s2o = t.sort()
    print('t', t)
    print('s', s)
    print('s2o', s2o)  # pass in index in sorted, get out index in original
    o2s = tensor_utils.invert_s2o(s2o)
    print('o2s', o2s)
    N = len(lengths)
    # following should return the original unsorted values essentially:
    for i in range(N):
        s_idx = o2s[i]
        print(s[s_idx].item())


def test_multi_argmax():
    t = torch.rand(7, 10)
    print('t', t)
    sizes = [3, 2, 5]
    res = tensor_utils.multi_argmax(sizes, t)
    print('res', res)


def test_multi_onehot():
    batch_size = 5
    sizes_l = [3, 2, 5]
    input = torch.LongTensor(batch_size, len(sizes_l))
    for i, size in enumerate(sizes_l):
        this_input = torch.from_numpy(np.random.choice(size, batch_size, replace=True))
        input[:, i] = this_input
    print('input', input)
    input_onehot = tensor_utils.multi_onehot(sizes_l=sizes_l, tensor=input)
    print('input_onehot', input_onehot)


def test_lengths_to_mask():
    N = 6
    max_len = 5
    lengths = torch.from_numpy(np.random.choice(max_len, N, replace=True)).long() + 1
    print('lengths', lengths)
    mask = tensor_utils.lengths_to_mask(lengths, max_len=max_len)
    print('mask', mask)


def test_masked_get():
    N = 3
    K = 4
    a = torch.rand(N, K)
    b = (torch.rand(N, K) > 0.5).bool()
    print('a', a)
    print('b', b)
    c = tensor_utils.masked_get(input=a, mask=b)
    print('c', c)


def test_hadamard():
    a = torch.rand(1, 3)
    b = torch.rand(1, 3)
    print('a', a)
    print('b', b)

    c = tensor_utils.Hadamard(a, b)
    print('c.size()', c.size())
    print('c', c)

    a = torch.rand(3)
    b = torch.rand(3)
    print('a', a)
    print('b', b)

    c = tensor_utils.Hadamard(a, b)
    print('c.size()', c.size())
    print('c', c)

    a = torch.rand(3, 1)
    b = torch.rand(3, 1)
    print('a', a)
    print('b', b)

    c = tensor_utils.Hadamard(a, b)
    print('c.size()', c.size())
    print('c', c)

    # a = torch.rand(3, 1)
    # b = torch.rand(3)
    # print('a', a)
    # print('b', b)

    # c = tensor_utils.Hadamard(a, b)
    # print('c.size()', c.size())
    # print('c', c)


def test_pack_sequences():
    N = 8
    max_len = 5
    V = 7
    lens = torch.from_numpy(np.random.choice(max_len, N, replace=True)) + 1
    seqs = torch.from_numpy(np.random.choice(V, (max_len, N), replace=True))
    print('')
    print('seqs\n', seqs)
    print('lens\n', lens)

    total_length = seqs.size(0)

    packed, reverse_idxes = tensor_utils.pack_sequences(seqs=seqs, lens=lens)
    print('packed', packed)

    seqs2, lens2 = tensor_utils.unpack_sequences(packed=packed, reverse_idxes=reverse_idxes, total_length=total_length)
    print('seqs2\n', seqs2)
    print('lens2\n', lens2)


def test_softmax_spatial_planar():
    N = 2
    C = 4
    H = 3
    W = 3
    inputs = torch.rand(N, C, H, W)
    print('inputs', inputs)
    print('inputs.sum(-1).sum(-1)', inputs.sum(-1).sum(-1))
    inputs2 = tensor_utils.softmax_spatial_planar(inputs)
    print('inputs2', inputs2)
    print('inputs2.sum(-1).sum(-1)', inputs2.sum(-1).sum(-1))


def test_params_diff():
    m1 = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4))
    m2 = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4))
    diff = tensor_utils.params_means_abs_diff(m1.parameters(), m2.parameters())
    print('diff %.3f' % diff)
    assert diff != 0

    m1b = copy.deepcopy(m1)
    diff = tensor_utils.params_means_abs_diff(m1.parameters(), m1b.parameters())
    print('diff %.3f' % diff)
    assert diff == 0


def test_named_params_diff():
    m1 = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4))
    m2 = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4))
    diff = tensor_utils.named_params_means_abs_diff(dict(m1.named_parameters()), dict(m2.named_parameters()))
    print('diff %.3f' % diff)
    assert diff != 0

    m1b = copy.deepcopy(m1)
    diff = tensor_utils.named_params_means_abs_diff(dict(m1.named_parameters()), dict(m1b.named_parameters()))
    print('diff %.3f' % diff)
    assert diff == 0


@pytest.mark.parametrize(
    "params1,params2,expected", [
        ([torch.Tensor([2, 3])], [torch.Tensor([3, 5])], (1 + 2) / 2),
        ([torch.Tensor([2, 3]), torch.Tensor([4.5])], [torch.Tensor([3, 5]), torch.Tensor([5.7])], (1 + 2 + 1.2) / 3),
    ]
)
def test_params_diff_parametrized(params1: Iterable[torch.Tensor], params2: Iterable[torch.Tensor], expected: float):
    diff = tensor_utils.params_means_abs_diff(params1, params2)
    assert torch.isclose(torch.Tensor([diff]), torch.Tensor([expected]))


@pytest.mark.parametrize(
    "params1,params2,expected", [
        ({'a': torch.Tensor([2, 3])}, {'a': torch.Tensor([3, 5])}, (1 + 2) / 2),
        (
            {'a': torch.Tensor([2, 3]), 'b': torch.Tensor([4.5])},
            {'a': torch.Tensor([3, 5]), 'b': torch.Tensor([5.7])},
            (1 + 2 + 1.2) / 3),
    ]
)
def test_named_params_diff_parametrized(
        params1: Dict[str, torch.Tensor], params2: Dict[str, torch.Tensor], expected: float):
    diff = tensor_utils.named_params_means_abs_diff(params1, params2)
    assert torch.isclose(torch.Tensor([diff]), torch.Tensor([expected]))


def test_idxes_to_onehot():
    idxes = torch.LongTensor([
        [3, 1, 0],
        [2, 2, 3]
    ])
    expected = torch.Tensor([
        [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ])
    onehot = tensor_utils.idxes_to_onehot(vocab_size=4, idxes=idxes)
    assert (expected == onehot).all()


def test_make_hard():
    target = torch.tensor([
        [0.5, 0.3, 0.7],
        [0.2, 0.4, 0.1]
    ], requires_grad=True)
    expected = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ])
    hard = tensor_utils.make_hard(target)
    assert (expected == hard).all()
    assert not hard.requires_grad
