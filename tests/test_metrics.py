from typing import Iterable

import torch
import numpy as np
import pytest

from ulfs import metrics


def test_tri_to_vec():
    # a = torch.rand(4, 4)
    # print('a', a)
    # print(metrics.tri_to_vec(a))
    # print(metrics.tri_to_vec(a))

    a = torch.Tensor([
        [2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17]
    ])
    vec1 = metrics.tri_to_vec(a)
    # vec2 = metrics.tri_to_vec(a)
    assert (vec1 == torch.Tensor([6, 10, 14, 11, 15, 16])).all()
    # assert (vec2 == torch.Tensor([2, 6, 10, 14, 7, 11, 15, 12, 16, 17])).all()


def test_topographic_similarity():
    N = 32
    K = 5
    M = 6
    a = torch.from_numpy(np.random.choice(K, (N, M), replace=True))
    b = torch.from_numpy(np.random.choice(K, (N, M), replace=True))
    rho = metrics.topographic_similarity(a, b)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, a)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(b, b)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, torch.cat([a[:N // 2], b[N // 2:]]))
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, a // 2)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, a // 4)
    print('rho %.3f' % rho)

    rho = metrics.topographic_similarity(a, torch.zeros(N, M).long())
    print('rho %.3f' % rho)


def test_uniqueness():
    N = 50
    K = 4
    V = 100
    a = torch.from_numpy(np.random.choice(V, (N, K), replace=True))
    print('uniqueness', metrics.uniqueness(a))
    a[:N // 5] = a[0]
    print('uniqueness', metrics.uniqueness(a))
    a[:] = a[0]
    print('uniqueness', metrics.uniqueness(a))


@pytest.mark.parametrize(
    'ground_truths,expected_clustering',
    [
        (['aa', 'bb', 'cc'], [0, 1, 2]),
        (['aa', 'bb', 'aa'], [0, 1, 0]),
    ]
)
def test_cluster_ground(ground_truths: Iterable[str], expected_clustering: Iterable[int]):
    pred_assignment = metrics.cluster_strings(ground_truths)
    assert (pred_assignment == torch.tensor(expected_clustering, dtype=torch.int64)).all()


@pytest.mark.parametrize(
    'utts,expected_clustering',
    [
        ([[0, 0], [1, 1], [2, 2]], [0, 1, 2]),
        ([[0, 0], [1, 1], [0, 0]], [0, 1, 0]),
        ([[1, 0], [2, 2], [2, 1], [1, 0]], [0, 1, 2, 0]),
    ]
)
def test_cluster_utts(utts: Iterable[Iterable[int]], expected_clustering: Iterable[int]):
    # utts_t_l = [torch.tensor(utt_l, dtype=torch.int64) for utt_l in utts]
    utts_t = torch.tensor(utts, dtype=torch.int64).transpose(0, 1).contiguous()
    pred_assignment = metrics.cluster_utts(utts_t)
    assert (pred_assignment == torch.tensor(expected_clustering, dtype=torch.int64)).all()


@pytest.mark.parametrize(
    "ground,pred,expected_prec,expected_rec",
    [
        ([0, 1, 2], [0, 0, 0], 0.5, 1.0),
        ([0, 0, 1], [0, 1, 2], 1.0, 0.75),
        ([0, 1, 2], [0, 0, 1], 0.75, 1.0),
        ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], 0.333, 1.0),
    ]
)
def test_cluster_prec_recall(
        ground: Iterable[int], pred: Iterable[int], expected_prec: float, expected_rec: float):
    precision, recall = metrics.calc_cluster_prec_recall(
        ground=torch.tensor(ground), pred=torch.tensor(pred))
    assert precision == pytest.approx(expected_prec, 0.01)
    assert recall == expected_rec
