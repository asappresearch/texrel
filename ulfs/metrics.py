from typing import List, Iterable, Dict, Tuple

import torch
import scipy.stats
import numpy as np


def lech_dist(A, B):
    """
    given two tensors A, and B, with the item index along the first dimension,
    and each tensor is 2-dimensional, this will calculate the lechenstein distance
    between each pair of examples between A and B

    both A and B are assumed to be long tensors of indices
    (cf one-hot)
    """
    assert A.dtype == torch.int64
    assert B.dtype == torch.int64

    N_a = A.size(0)
    N_b = B.size(0)
    E = A.size(1)
    assert E == B.size(1)
    assert len(A.size()) == 2
    assert len(B.size()) == 2
    if N_a * N_b * 4 / 1000 / 1000 >= 500:  # if use > 500MB memory, then die
        raise Exception('Would use too much memory => dieing')
    A = A.unsqueeze(1).expand(N_a, N_b, E)
    B = B.unsqueeze(0).expand(N_a, N_b, E)
    AeqB = A == B
    dists = AeqB.sum(dim=-1)
    dists = dists.float() / E

    return dists


def lech_dist_from_samples(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """
    left and right are two sets of samples from the same
    tensor. they should both be two dimensional. dim 0
    is the sample index. dim 1 is the dimension we will
    calculate lechenstein distances over

    both left and right are assumed to be long tensors of indices
    (cf one-hot)
    """
    assert left.dtype == torch.int64
    assert right.dtype == torch.int64

    assert len(left.size()) == 2
    assert len(right.size()) == 2

    N = left.size(0)
    assert right.size(0) == N
    E = left.size(1)
    assert E == right.size(1)

    left_eq_right = left == right
    dists = left_eq_right.sum(dim=-1)
    dists = dists.float() / E

    return dists


def tri_to_vec(tri):
    """
    returns lower triangle of a square matrix, as a vector, excluding the diagonal

    eg given

    1 3 9
    4 3 7
    2 1 5

    returns:

    4 2 1
    """
    assert len(tri.size()) == 2
    assert tri.size(0) == tri.size(1)
    K = tri.size(0)
    res_size = (K - 1) * K // 2
    res = torch.zeros(res_size, dtype=tri.dtype)
    pos = 0
    for k in range(K - 1):
        res[pos:pos + (K - k - 1)] = tri[k + 1:, k]
        pos += (K - k - 1)
    return res


def calc_squared_euc_dist(one, two):
    """
    input: two arrays, [N1][E]
                       [N2][E]
    output: one matrix: [N1][N2]
    """
    one_squared = (one * one).sum(dim=1)
    two_squared = (two * two).sum(dim=1)
    transpose = one @ two.transpose(0, 1)
    squared_dist = one_squared.unsqueeze(1) + two_squared.unsqueeze(0) - 2 * transpose
    return squared_dist


def get_pair_idxes(length, max_samples):
    """
    return pairs of indices
    each pair should be unique
    no more than max_samples pairs will be returned
    (will sample if length * length > max_samples)
    """
    if length * length <= max_samples:
        idxes = torch.ones(length * length, 2, dtype=torch.int64)
        idxes = idxes.cumsum(dim=0) - 1
        idxes[:, 0] = idxes[:, 0] // length
        idxes[:, 1] = idxes[:, 1] % length
    else:
        # we sample with replacement, assuming number of
        # possible pairs >> max_samples
        a_idxes = torch.from_numpy(np.random.choice(
            length, max_samples, replace=True))
        b_idxes = torch.from_numpy(np.random.choice(
            length, max_samples, replace=True))
        idxes = torch.stack([a_idxes, b_idxes], dim=1)
    return idxes


def topographic_similarity(utts: torch.Tensor, labels: torch.Tensor, max_samples=10000):
    """
    (quoting Angeliki 2018)
    "The intuition behind this measure is that semantically similar objects should have similar messages."

    a and b should be discrete; 2-dimensional. with item index along first dimension, and attribute index
    along second dimension

    if there are more pairs of utts and labels than max_samples, then sample pairs

    Parameters
    ----------
    utts: torch.Tensor
        assumed to be long tensor
    labels: torch.Tensor
        assumed to be long tensor
    """
    assert utts.size(0) == labels.size(0)
    assert len(utts.size()) == 2
    assert len(labels.size()) == 2
    assert utts.dtype == torch.int64
    assert labels.dtype == torch.int64

    pair_idxes = get_pair_idxes(utts.size(0), max_samples=max_samples)

    utts_left, utts_right = utts[pair_idxes[:, 0]], utts[pair_idxes[:, 1]]
    labels_left, labels_right = labels[pair_idxes[:, 0]], labels[pair_idxes[:, 1]]

    utts_pairwise_dist = lech_dist_from_samples(utts_left, utts_right)
    labels_pairwise_dist = lech_dist_from_samples(labels_left, labels_right)

    rho, _ = scipy.stats.spearmanr(a=utts_pairwise_dist.cpu(), b=labels_pairwise_dist.cpu())
    if rho != rho:
        # if rho is nan, we'll assume taht utts was all the same value. hence rho
        # is zero. (if labels was all the same value too, rho would be unclear, but
        # since the labels are provided by the dataset, we'll assume that they are diverse)
        max_utts_diff = (utts_pairwise_dist - utts_pairwise_dist[0]).abs().max().item()
        max_labels_diff = (labels_pairwise_dist - labels_pairwise_dist[0]).abs().max().item()
        print('rho is zero, max_utts_diff', max_utts_diff, 'max_labels_diff', max_labels_diff)
        rho = 0
    return rho


def uniqueness(a: torch.Tensor) -> float:
    """
    given 2 dimensional discrete tensor a, will count the number of unique vectors,
    and divide by the total number of vectors, ie returns the fraction of vectors
    which are unique

    Parameters
    ----------
    a: torch.Tensor
        should be long tensor of indices
        should be dimensions [N][K]
    """
    assert a.dtype == torch.int64
    v = set()
    if len(a.size()) != 2:
        raise ValueError('size of a should be 2-dimensions, but a.size() is ' + str(a.size()))
    N, K = a.size()
    for n in range(N):
        v.add(','.join([str(x) for x in a[n].tolist()]))
    return (len(v) - 1) / (N - 1)   # subtract 1, because if everything is identical, there would still be 1


def cluster_strings(strings: Iterable[str]) -> torch.Tensor:
    """
    given a list of strings, assigns a clustering, where
    each pair of identical ground truth strings is in the same
    cluster
    return a torch.LongTensor containing the cluster id of
    each ground truth
    """
    cluster_id_by_truth: Dict[str, int] = {}
    cluster_l: List[int] = []
    for n, truth in enumerate(strings):
        cluster_id = cluster_id_by_truth.setdefault(truth, len(cluster_id_by_truth))
        cluster_l.append(cluster_id)
    return torch.tensor(cluster_l, dtype=torch.int64)


def cluster_utts(utts: torch.Tensor) -> torch.Tensor:
    """
    given a 2-d tensor of [S][N], where N is number of
    examples, and S is sequence length, and the tensor
    is of discrete int64 indices (cf distributions over
    tokens), we cluster all identical examples, and return
    a cluster assignment as a long tensor, containing the
    cluster id of each example, starting from 0

    if examples have differnet lengths, padding id should
    be identical. this function will compare the entire
    length of each example. as long as the padding id is
    consistent, this should work as desired, i.e. effectively
    ignore padding
    """
    S, N = utts.size()
    clustering = torch.zeros(N, dtype=torch.int64)
    seen = torch.zeros(N, dtype=torch.bool)
    cluster_id = 0
    for n in range(N):
        if seen[n]:
            continue
        mask = (utts == utts[:, n:n + 1]).all(dim=0)
        clustering[mask] = cluster_id
        cluster_id += 1
        seen[mask] = True
    return clustering


def calc_cluster_prec_recall(pred: torch.Tensor, ground: torch.Tensor) -> Tuple[float, float]:
    """
    given predicted clustering, and ground clustering,
    calculates cluster recall and precision
    """
    assert len(pred.size()) == 1
    assert len(ground.size()) == 1

    N = ground.size(0)
    assert pred.size(0) == N

    left_indices = torch.ones(N, dtype=torch.int64).cumsum(dim=0) - 1
    right_indices = torch.ones(N, dtype=torch.int64).cumsum(dim=0) - 1

    left_indices = left_indices.unsqueeze(-1).expand(N, N)
    right_indices = right_indices.unsqueeze(0).expand(N, N)

    left_indices = left_indices.contiguous().view(-1)
    right_indices = right_indices.contiguous().view(-1)

    dup_mask = left_indices <= right_indices

    ground_pos = ground[left_indices] == ground[right_indices]
    pred_pos = pred[left_indices] == pred[right_indices]

    tp = ((pred_pos & ground_pos) & dup_mask).sum().item()
    fp = ((pred_pos & (~ground_pos)) & dup_mask).sum().item()
    fn = (((~pred_pos) & ground_pos) & dup_mask).sum().item()
    tn = (((~pred_pos) & (~ground_pos)) & dup_mask).sum().item()

    assert tp + fp + fn + tn == N * (N + 1) / 2

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall
