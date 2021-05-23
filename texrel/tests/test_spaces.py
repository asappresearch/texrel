import math

import pytest
import numpy as np

from texrel import spaces


def test_things():
    r = np.random.RandomState()
    thing_space = spaces.ThingSpace(
        r=r,
        shape_space=spaces.IntSpace(r=r, num_ints=9), color_space=spaces.IntSpace(r=r, num_ints=9))
    for i in range(5):
        print(thing_space.sample())
    print('')

    thing_space = spaces.ThingSpace(
        r=r,
        shape_space=spaces.IntSpace(r=r, num_ints=2),
        color_space=spaces.IntSpace(r=r, num_ints=2)
    )
    for i in range(5):
        print(thing_space.sample())


def test_int_space_partition():
    r = np.random.RandomState()
    orig_size = 9
    int_space = spaces.IntSpace(r=r, num_ints=orig_size)
    print('len(int_space)', len(int_space))

    r = np.random.RandomState()

    partitions = [len(int_space) - 3, 3]
    spaces_l = int_space.partition(sizes=partitions)
    assert sum([len(space) for space in spaces_l]) == orig_size
    for i, space in enumerate(spaces_l):
        print(space)
        for j, space2 in enumerate(spaces_l):
            if i == j:
                continue
            assert len(set(space2.avail_ints) & set(space.avail_ints)) == 0


@pytest.mark.parametrize(
    "num_dims,num_ints",
    [
        (3, 9),
        (5, 9)
    ]
)
def test_multi_int_unordered_space_partition(num_dims, num_ints):
    r = np.random.RandomState()
    int_space = spaces.MultiIntUnorderedSpace(num_dims=num_dims, num_ints=num_ints, r=r)
    len_full_space = len(int_space)
    print('len(int_space)', len_full_space)
    assert len(int_space) == math.factorial(num_ints + num_dims - 1) // math.factorial(
        num_dims) // math.factorial(num_ints - 1)

    partitions = [len(int_space) - 3, 3]
    spaces_l = int_space.partition(sizes=partitions)
    assert sum([len(space) for space in spaces_l]) == len_full_space
    for i, space in enumerate(spaces_l):
        print(space)
        for j, space2 in enumerate(spaces_l):
            if i == j:
                continue
            assert len(set(space2.avail_ints) & set(space.avail_ints)) == 0


@pytest.mark.parametrize(
    "num_dims,num_ints",
    [
        (3, 9),
        (5, 9)
    ]
)
def test_multi_int_space_resample_dim(num_dims: int, num_ints: int):
    r = np.random.RandomState()

    int_space = spaces.MultiIntUnorderedSpace(num_dims=num_dims, num_ints=num_ints, r=r)
    len_full_space = len(int_space)
    print('len(int_space)', len_full_space)
    # if dims == [9, 9, 9]:
    #     num_ints = 9
    #     num_dims = 3
    assert len(int_space) == math.factorial(num_ints + num_dims - 1) // math.factorial(
        num_dims) // math.factorial(num_ints - 1)

    partitions = [len(int_space) - 3, 3]
    spaces_l = int_space.partition(sizes=partitions)
    assert sum([len(space) for space in spaces_l]) == len_full_space
    for i, space in enumerate(spaces_l):
        print(space)
        for j, space2 in enumerate(spaces_l):
            if i == j:
                continue
            assert len(set(space2.avail_ints) & set(space.avail_ints)) == 0
    train_space, holdout_space = spaces_l
    print('train_space', train_space)
    for it in range(20):
        ints = train_space.sample()
        print('ints', ints)
        assert ints in train_space
        assert ints not in holdout_space
        resample_dim = np.random.randint(num_dims)
        print('resample_dim', resample_dim)
        resampled = train_space.resample_dim(ints=ints, dim=resample_dim)
        print('    resampled', resampled)
        assert resampled in train_space
        assert resampled not in holdout_space  # get evidence that `in` is working...


def test_thing_space_partition():
    r = np.random.RandomState()
    thing_space = spaces.ThingSpace(
        r=r,
        shape_space=spaces.IntSpace(r=r, num_ints=9),
        color_space=spaces.IntSpace(r=r, num_ints=9))
    num_items = thing_space.num_unique_things
    print('num_items', num_items)

    partitions = [num_items - 5, 5]
    spaces_ = thing_space.partition(partitions)
    samples_l = []
    for i, s in enumerate(spaces_):
        print('i', i, 'size', s.num_unique_things)
        for k in range(5):
            print(s.sample())
        samples = set()
        for i in range(10000):
            samples.add(s.sample())
        print('len(samples)', len(samples))
        print('samples[:5]', list(samples)[:5])
        samples_l.append(samples)
    for i, s in enumerate(samples_l):
        print('len(samples)', len(s))
        others = set()
        for j, s2 in enumerate(samples_l):
            if i != j:
                others |= s2
        print('len(others)', len(others))
        for o in s:
            if o in others:
                raise Exception("intersection")
