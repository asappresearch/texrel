import torch

import numpy as np

from texrel import things, spaces


def test_shape_color_as_english():
    shape = things.ShapeColor(color=3, shape=2)
    assert shape.as_english() == 'color3 shape2'


def test_shape_color_as_english_structure():
    shape = things.ShapeColor(color=3, shape=2)
    assert shape.as_english_structure() == ('color3', 'shape2')


def test_things_encode_decode():
    r = np.random.RandomState()
    thing_space = spaces.ThingSpace(
        r=r,
        shape_space=spaces.IntSpace(r=r, num_ints=9),
        color_space=spaces.IntSpace(r=r, num_ints=9)
    )
    for i in range(10):
        o = thing_space.sample()
        o_indices = o.as_onehot_indices(thing_space)
        o_onehot = torch.Tensor(o.as_onehot_tensor_size(thing_space)).zero_()
        o_onehot[torch.LongTensor(o_indices)] = 1
        o2 = things.ShapeColor.from_onehot_tensor(thing_space, o_onehot)
        print('o', o, 'o2', o2)
        assert o2 == o


def test_things_eat():
    r = np.random.RandomState()
    thing_space = spaces.ThingSpace(
        r=r,
        shape_space=spaces.IntSpace(r=r, num_ints=9),
        color_space=spaces.IntSpace(r=r, num_ints=9)
    )
    o_sample = thing_space.sample()
    thing_onehot_size = o_sample.as_onehot_tensor_size(thing_space)
    print('thing_onehot_size', thing_onehot_size)
    for i in range(10):
        o1 = thing_space.sample()
        o2 = thing_space.sample()
        if o1 == o2:
            continue
        assert o1 != o2

        o_t = torch.Tensor(thing_onehot_size * 2).zero_()

        o1_indices = o1.as_onehot_indices(thing_space)
        o2_indices = o2.as_onehot_indices(thing_space)
        o_t[torch.LongTensor(o1_indices)] = 1
        o_t[torch.LongTensor(o2_indices) + thing_onehot_size] = 1
        o1b, o_t = things.ShapeColor.eat_from_onehot_tensor(thing_space, o_t)
        o2b, o_t = things.ShapeColor.eat_from_onehot_tensor(thing_space, o_t)
        assert o_t.size()[0] == 0

        print('o1', o1, 'o1b', o1b, 'o2', o2, 'o2b', o2b)
        assert o1 == o1b
        assert o2 == o2b
        assert o1 != o2
        assert o1b != o2b
