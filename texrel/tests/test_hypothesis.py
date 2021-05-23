"""
test ability to generate hypothesis
"""
from typing import Tuple, Any, Union, List
import pytest
import mock

from texrel import hypothesis, things, relations, spaces

import numpy as np


def create_multi_int_unordered_hip(ints: List[int]):
    return hypothesis.MultiIntUnorderedHypothesis(r=np.random.RandomState(), ints=tuple(ints), dim_length=9)


def create_multi_int_unordered_hip_from_sc_pairs(sc_pairs: List[Tuple[int, int]]):
    sc_ints = [s * 9 + c for s, c in sc_pairs]
    return hypothesis.MultiIntUnorderedHypothesis(r=np.random.RandomState(), ints=tuple(sc_ints), dim_length=9)


@pytest.mark.parametrize(
    "colors",
    [
        [2],
        [3, 4],
        [4, 8, 3]
    ]
)
def test_colors_hypothesis(colors):
    r = np.random.RandomState()
    multi_int_unordered_hyp = create_multi_int_unordered_hip(colors)
    h = hypothesis.ColorsHypothesis(
        multi_int_unordered_hyp=multi_int_unordered_hyp, shape_space=spaces.IntSpace(r=r, num_ints=9))
    print('h', h)

    h_english = h.as_english()
    print('multi_int_unordered_hyp.dim_length', multi_int_unordered_hyp.dim_length)
    h2 = hypothesis.ColorsHypothesis.from_english(
        r=r, hyp_eng=h_english, num_colors=multi_int_unordered_hyp.dim_length,
        shape_space=h.shape_space
    )
    print('h2', h2)
    assert h == h2

    for num_distractors in range(3):
        print('')
        print('=================')
        print('distractors=', num_distractors)
        print('pos examples:')
        for i in range(3):
            grid = h.create_positive_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')
            assert len(grid.objects_set) == len(colors) + num_distractors
            to_match = list(colors)
            for o in grid.objects_set:
                if o.color in to_match:
                    to_match.remove(o.color)
            assert len(to_match) == 0
        print('')
        print('neg examples:')
        for i in range(3):
            grid = h.create_negative_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')
            assert len(grid.objects_set) == len(colors) + num_distractors
            to_match = list(colors)
            for o in grid.objects_set:
                if o.color in to_match:
                    to_match.remove(o.color)
            assert len(to_match) == 1  # we want tight/hard negative examples


@pytest.mark.parametrize(
    "shapes",
    [
        [2],
        [3, 4],
        [4, 8, 3]
    ]
)
def test_shapes_hypothesis(shapes):
    r = np.random.RandomState()
    multi_int_unordered_hyp = create_multi_int_unordered_hip(shapes)
    h = hypothesis.ShapesHypothesis(
        multi_int_unordered_hyp=multi_int_unordered_hyp, color_space=spaces.IntSpace(r=r, num_ints=9))
    print('h', h)

    for num_distractors in range(3):
        print('')
        print('=================')
        print('distractors=', num_distractors)
        print('pos examples:')
        for i in range(100):
            grid = h.create_positive_example(num_distractors=num_distractors, grid_size=5)
            if i < 3:
                print(grid)
                print('')
            assert len(grid.objects_set) == len(shapes) + num_distractors
            to_match = list(shapes)
            for o in grid.objects_set:
                if o.shape in to_match:
                    to_match.remove(o.shape)
            assert len(to_match) == 0
        print('')
        print('neg examples:')
        for i in range(100):
            grid = h.create_negative_example(num_distractors=num_distractors, grid_size=5)
            if i < 3:
                print(grid)
                print('')
            assert len(grid.objects_set) == len(shapes) + num_distractors
            to_match = list(shapes)
            for o in grid.objects_set:
                if o.shape in to_match:
                    to_match.remove(o.shape)
            assert len(to_match) == 1  # we want tight/hard negative examples


@pytest.mark.parametrize(
    "things",
    [
        [things.ShapeColor(shape=3, color=4)],
        [things.ShapeColor(shape=3, color=4), things.ShapeColor(shape=5, color=7)],
        [things.ShapeColor(shape=3, color=4), things.ShapeColor(shape=5, color=7),
         things.ShapeColor(shape=2, color=1)],
    ]
)
def test_things_hypothesis(things: List[things.ShapeColor]):
    num_colors = 9
    num_shapes = 9
    r = np.random.RandomState()
    all_thing_space = spaces.ThingSpace(
        r=r,
        shape_space=spaces.IntSpace(r=r, num_ints=num_shapes),
        color_space=spaces.IntSpace(r=r, num_ints=num_colors))
    h = hypothesis.ThingsHypothesis(
        r=r, things=things, thing_space=all_thing_space, distractor_thing_space=all_thing_space)
    print('h', h)

    for num_distractors in range(3):
        print('')
        print('=================')
        print('distractors=', num_distractors)
        print('pos examples:')
        for i in range(100):
            grid = h.create_positive_example(num_distractors=num_distractors, grid_size=5)
            if i < 3:
                print(grid)
                print('')
            assert len(grid.objects_set) == len(things) + num_distractors
            to_match = list(things)
            for o in grid.objects_set:
                if o in to_match:
                    to_match.remove(o)
            assert len(to_match) == 0
        print('')
        print('neg examples:')
        for i in range(100):
            grid = h.create_negative_example(num_distractors=num_distractors, grid_size=5)
            if i < 3:
                print(grid)
                print('')
            assert len(grid.objects_set) == len(things) + num_distractors
            to_match = list(things)
            for o in grid.objects_set:
                if o in to_match:
                    to_match.remove(o)
            assert len(to_match) == 1  # we want tight/hard negative examples


@pytest.mark.parametrize(
    "h,english,english_structure",
    [
        (hypothesis.ColorsHypothesis(
            multi_int_unordered_hyp=create_multi_int_unordered_hip([4]), shape_space=mock.Mock()),
            'has-colors color4', ('has-colors', 'color4')),
        (hypothesis.ColorsHypothesis(
            multi_int_unordered_hyp=create_multi_int_unordered_hip([4, 7]), shape_space=mock.Mock()),
            'has-colors color4 color7', ('has-colors', ('color4', 'color7'))),
        (hypothesis.ShapesHypothesis(
            multi_int_unordered_hyp=create_multi_int_unordered_hip([4]), color_space=mock.Mock()),
            'has-shapes shape4', ('has-shapes', 'shape4')),
        (hypothesis.ShapesHypothesis(
            multi_int_unordered_hyp=create_multi_int_unordered_hip([4, 7]), color_space=mock.Mock()),
            'has-shapes shape4 shape7', ('has-shapes', ('shape4', 'shape7'))),
        (hypothesis.ThingsHypothesis(
            r=np.random.RandomState(),
            things=[things.ShapeColor(shape=4, color=3)],
            thing_space=mock.Mock(),
            distractor_thing_space=mock.Mock()),
            'has-shapecolors color3 shape4', ('has-shapecolors', ('color3', 'shape4'))),
        (hypothesis.RelationHypothesis(
            r=np.random.RandomState(),
            relation=relations.Relation(
                left=things.ShapeColor(color=2, shape=3),
                prep=relations.RightOf(),
                right=things.ShapeColor(color=4, shape=5)),
            rel_space=mock.Mock(),
            distractor_thing_space=mock.Mock()),
            'color2 shape3 right-of color4 shape5',
            ('right-of', (('color2', 'shape3'), ('color4', 'shape5')))),
    ]
)
def test_english_structure(h: hypothesis.Hypothesis, english: str, english_structure: Union[str, Tuple[Any, Any]]):
    print(h.as_english())
    print(h.as_english_structure())
    assert h.as_english() == english
    assert h.as_english_structure() == english_structure
