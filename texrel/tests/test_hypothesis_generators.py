from typing import List, Dict
from collections import Counter

import pytest
import numpy as np

from texrel import hypothesis_generators as hg, hypothesis, relations


@pytest.mark.parametrize(
    "ints,dim_length,num_distractors", [
        ([1, 3, 5], 9, 2),
        ([3, 1, 1], 4, 2),
    ]
)
def test_multi_int_unordered_hypothesis(ints: List[int], dim_length: int, num_distractors: int):
    r = np.random.RandomState()
    generator = hypothesis.MultiIntUnorderedHypothesis(r=r, ints=tuple(ints), dim_length=dim_length)
    for it in range(10):
        ex = generator.create_positive_example(num_distractors=num_distractors)
        print('pos ex', ex)
        assert len(ex) == len(ints) + num_distractors
        for v in ints:
            assert v in ex

    for it in range(10):
        ex = generator.create_negative_example(num_distractors=num_distractors)
        print('neg ex', ex)
        assert len(ex) == len(ints) + num_distractors
        # check cannot find the example ints in resulting ints
        to_match = list(ints)
        for v in ex:
            if v in to_match:
                to_match.remove(v)
        assert len(to_match) > 0


@pytest.mark.parametrize(
    "num_dims,dim_length,num_distractors,num_holdout", [
        (1, 3, 2, 1),
        (2, 4, 2, 2),
        (2, 9, 2, 4),
        (3, 9, 2, 3),
        (4, 9, 2, 4),
    ]
)
def test_multi_int_unordered_generator_base(
        num_dims: int, dim_length: int, num_distractors: int, num_holdout: int):
    r = np.random.RandomState()
    generator = hg.MultiIntUnorderedGenerator(
        num_dims=num_dims, dim_length=dim_length, r=r, num_holdout=num_holdout)
    print('generator.training_space', generator.train_space)
    print('generator.holdout_space', generator.holdout_space)
    for train in [True, False]:
        print('train', train)
        count_by_num_distinct_ints: Dict[int, int] = Counter()
        for it in range(1000):
            h = generator.sample_hyp(train=train)
            if it < 3:
                print('h', h)
            assert len(h.ints) == num_dims
            if train:
                assert h.ints in generator.train_space
                assert h.ints not in generator.holdout_space
            else:
                assert h.ints not in generator.train_space
                assert h.ints in generator.holdout_space
            num_distinct_ints = len(set(h.ints))
            if it < 3:
                print('h.ints', h.ints, 'num distinct ints', num_distinct_ints)
            count_by_num_distinct_ints[num_distinct_ints] += 1

            ex = h.create_positive_example(num_distractors=2)
            if it < 3:
                print('pos ex', ex)

            ex = h.create_negative_example(num_distractors=2)
            if it < 3:
                print('neg ex', ex)
        print('count_by_num_distinct_ints', count_by_num_distinct_ints)


@pytest.mark.parametrize(
    "num_dims,dim_length,num_distractors,num_holdout", [
        (1, 3, 2, 1),
        (2, 9, 2, 2),
        (3, 9, 2, 3),
        (3, 9, 2, 3),
        (4, 9, 2, 4),
    ]
)
def test_multi_int_unordered_generator_seed(
        num_dims: int, dim_length: int, num_distractors: int, num_holdout: int):
    r1 = np.random.RandomState(123)
    r2 = np.random.RandomState(123)
    generator1 = hg.MultiIntUnorderedGenerator(
        num_dims=num_dims, dim_length=dim_length, r=r1, num_holdout=num_holdout)
    generator2 = hg.MultiIntUnorderedGenerator(
        num_dims=num_dims, dim_length=dim_length, r=r2, num_holdout=num_holdout)
    for it in range(100):
        h1 = generator1.sample_hyp(train=True)
        h2 = generator2.sample_hyp(train=True)
        assert h1.ints == h2.ints
        if it < 3:
            print('h1, train', h1, h2)

        h1 = generator1.sample_hyp(train=False)
        h2 = generator2.sample_hyp(train=False)
        assert h1.ints == h2.ints
        if it < 3:
            print('h, holdout', h1, h2)

        ex1 = h1.create_positive_example(num_distractors=2)
        ex2 = h2.create_positive_example(num_distractors=2)
        assert ex1 == ex2
        if it < 3:
            print('pos ex', ex1, ex2)

        ex1 = h1.create_negative_example(num_distractors=2)
        ex2 = h2.create_negative_example(num_distractors=2)
        assert ex1 == ex2
        if it < 3:
            print('neg ex', ex1, ex2)


@pytest.mark.parametrize(
    "num_entities,num_colors,num_shapes,num_holdout",
    [
        (1, 9, 7, 4),
        (2, 9, 7, 4),
        (3, 9, 7, 4),
    ]
)
def test_colors_hg(num_entities: int, num_colors: int, num_shapes: int, num_holdout: int):
    num_distractors = 2

    r = np.random.RandomState()
    generator = hg.ColorsHG(
        r=r, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    for train in [True, False]:
        print('train', train)
        for it in range(3):
            h: hypothesis.ColorsHypothesis = generator.sample_hyp(train=train)  # type: ignore
            print('h', h)
            assert len(h.colors) == num_entities
            # for c in h.colors:
            print('h.colors', h.colors)
            if train:
                assert h.colors not in generator.multi_int_unordered_generator.holdout_space
                assert h.colors in generator.multi_int_unordered_generator.train_space
            else:
                assert h.colors in generator.multi_int_unordered_generator.holdout_space
                assert h.colors not in generator.multi_int_unordered_generator.train_space
            for it in range(3):
                ex = h.create_positive_example(grid_size=4, num_distractors=num_distractors)
                print('pos ex\n' + str(ex))
                assert len(ex.objects_set) == num_entities + num_distractors
            for it in range(3):
                ex = h.create_negative_example(grid_size=4, num_distractors=num_distractors)
                print('neg ex\n' + str(ex))
                assert len(ex.objects_set) == num_entities + num_distractors


@pytest.mark.parametrize(
    "num_entities,num_colors,num_shapes,num_holdout",
    [
        (1, 9, 7, 4),
        (2, 9, 7, 4),
        (3, 9, 7, 4),
    ]
)
def test_colors_hg_seed(num_entities: int, num_colors: int, num_shapes: int, num_holdout: int):
    num_distractors = 2

    r1 = np.random.RandomState(123)
    r2 = np.random.RandomState(123)
    generator1 = hg.ColorsHG(
        r=r1, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    generator2 = hg.ColorsHG(
        r=r2, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    for it in range(3):
        h1: hypothesis.ColorsHypothesis = generator1.sample_hyp(train=True)  # type: ignore
        h2: hypothesis.ColorsHypothesis = generator2.sample_hyp(train=True)  # type: ignore
        assert h1.colors == h2.colors
        print('h', h1, h2)
        for it in range(3):
            ex1 = h1.create_positive_example(grid_size=4, num_distractors=num_distractors)
            ex2 = h2.create_positive_example(grid_size=4, num_distractors=num_distractors)
            print('pos ex\n' + str(ex1) + " " + str(ex2))
            assert ex1 == ex2
        for it in range(3):
            ex1 = h1.create_negative_example(grid_size=4, num_distractors=num_distractors)
            ex2 = h2.create_negative_example(grid_size=4, num_distractors=num_distractors)
            print('neg ex\n' + str(ex1) + " " + str(ex2))
            assert ex1 == ex2


@pytest.mark.parametrize(
    "num_entities,num_colors,num_shapes,num_holdout",
    [
        (1, 9, 7, 3),
        (2, 9, 7, 3),
        (3, 9, 7, 3),
    ]
)
def test_shapes_hg(num_entities: int, num_colors: int, num_shapes: int, num_holdout: int):
    num_distractors = 2

    r = np.random.RandomState()
    generator = hg.ShapesHG(
        r=r, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    for it in range(3):
        h: hypothesis.ShapesHypothesis = generator.sample_hyp(train=True)  # type: ignore
        print('h', h)
        assert len(h.shapes) == num_entities
        for it in range(3):
            ex = h.create_positive_example(grid_size=4, num_distractors=num_distractors)
            print('pos ex\n' + str(ex))
            assert len(ex.objects_set) == num_entities + num_distractors
        for it in range(3):
            ex = h.create_negative_example(grid_size=4, num_distractors=num_distractors)
            print('neg ex\n' + str(ex))
            assert len(ex.objects_set) == num_entities + num_distractors


@pytest.mark.parametrize(
    "num_entities,num_colors,num_shapes,num_holdout",
    [
        (1, 9, 7, 3),
        (2, 9, 7, 3),
        (3, 9, 7, 3),
    ]
)
def test_shapes_hg_seed(num_entities: int, num_colors: int, num_shapes: int, num_holdout: int):
    num_distractors = 2

    r1 = np.random.RandomState(123)
    r2 = np.random.RandomState(123)
    generator1 = hg.ShapesHG(
        r=r1, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    generator2 = hg.ShapesHG(
        r=r2, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    assert generator1.color_space.r != generator2.color_space.r
    assert generator1.color_space.r == r1
    assert generator2.color_space.r == r2
    for it in range(10):
        h1: hypothesis.ShapesHypothesis = generator1.sample_hyp(train=True)  # type: ignore
        h2: hypothesis.ShapesHypothesis = generator2.sample_hyp(train=True)  # type: ignore
        assert h1.r == r1
        assert h2.r == r2
        assert h1.shapes == h2.shapes
        for j in range(10):
            ex1 = h1.create_positive_example(grid_size=4, num_distractors=num_distractors)
            ex2 = h2.create_positive_example(grid_size=4, num_distractors=num_distractors)
            if it == 0 and j < 3:
                print('pos ex\n' + str(ex1) + ' ' + str(ex2))
            assert ex1 == ex2
        for j in range(10):
            ex1 = h1.create_negative_example(grid_size=4, num_distractors=num_distractors)
            ex2 = h2.create_negative_example(grid_size=4, num_distractors=num_distractors)
            if it == 0 and j < 3:
                print('neg ex\n' + str(ex1) + ' ' + str(ex2))
            assert ex2 == ex2


@pytest.mark.parametrize(
    "num_entities,num_colors,num_shapes,num_holdout,num_distractors",
    [
        (1, 3, 3, 1, 0),
        (1, 3, 3, 1, 2),
        (2, 3, 3, 1, 2),
        (3, 3, 3, 1, 2),
        (1, 9, 7, 1, 2),
        (1, 9, 7, 5, 2),
        (2, 9, 7, 1, 2),
        (2, 9, 7, 5, 2),
        (3, 9, 7, 1, 2),
        (3, 9, 7, 5, 2),
    ]
)
def test_things_hg_base(
        num_entities: int, num_colors: int, num_shapes: int, num_holdout: int,
        num_distractors: int):
    r = np.random.RandomState()
    generator = hg.ThingsHG(
        r=r, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    train_space = generator.thing_space_training
    holdout_space = generator.thing_space_holdout

    print('train space', train_space)
    print('holdout space', holdout_space)
    for train in [True, False]:
        print('train', train)
        for it in range(100):
            h: hypothesis.ThingsHypothesis = generator.sample_hyp(train=train)  # type: ignore
            if it < 3:
                print('h', h)
            assert len(h.things) == num_entities
            for j in range(3):
                ex = h.create_positive_example(grid_size=4, num_distractors=num_distractors)
                if it < 3 and j < 3:
                    print('pos ex\n' + str(ex))
                assert len(ex.objects_set) == num_entities + num_distractors
            for j in range(3):
                ex = h.create_negative_example(grid_size=4, num_distractors=num_distractors)
                if it < 3 and j < 3:
                    print('neg ex\n' + str(ex))
                assert len(ex.objects_set) == num_entities + num_distractors


@pytest.mark.parametrize(
    "num_entities,num_colors,num_shapes,num_holdout",
    [
        (1, 9, 7, 5),
        (2, 9, 7, 5),
        (3, 9, 7, 5),
    ]
)
def test_things_hg_seed(num_entities: int, num_colors: int, num_shapes: int, num_holdout: int):
    num_distractors = 2

    r1 = np.random.RandomState(123)
    r2 = np.random.RandomState(123)
    generator1 = hg.ThingsHG(
        r=r1, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    generator2 = hg.ThingsHG(
        r=r2, num_entities=num_entities, num_avail_colors=num_colors, num_holdout=num_holdout,
        num_avail_shapes=num_shapes)
    for it in range(3):
        h1: hypothesis.ThingsHypothesis = generator1.sample_hyp(train=True)  # type: ignore
        h2: hypothesis.ThingsHypothesis = generator2.sample_hyp(train=True)  # type: ignore
        assert h1.things == h2.things
        for j in range(10):
            ex1 = h1.create_positive_example(grid_size=4, num_distractors=num_distractors)
            ex2 = h2.create_positive_example(grid_size=4, num_distractors=num_distractors)
            if it == 0 and j < 3:
                print('pos ex\n' + str(ex1) + ' ' + str(ex2))
            assert ex1 == ex2
        for j in range(3):
            ex1 = h1.create_negative_example(grid_size=4, num_distractors=num_distractors)
            ex2 = h2.create_negative_example(grid_size=4, num_distractors=num_distractors)
            if it == 0 and j < 3:
                print('neg ex\n' + str(ex1) + ' ' + str(ex2))
            assert ex1 == ex2


def test_relation_hg():
    r = np.random.RandomState()
    num_holdout = 3
    generator = hg.RelationsHG(
        r=r, available_preps=['LeftOf', 'Above'],
        num_avail_colors=9, num_avail_shapes=9, num_holdout=num_holdout)
    for train in [True, False]:
        relation = generator.sample_hyp(train=train)
        print('relation', relation)

        prep = relation.relation.prep
        left = relation.relation.left
        right = relation.relation.right

        for num_distractors in range(3):
            print('')
            print('=================')
            print('distractors=', num_distractors)
            print('pos examples:')
            for i in range(100):
                grid = relation.create_positive_example(num_distractors=num_distractors, grid_size=5)
                if i < 3:
                    print(grid)
                    print('')
                assert len(grid.objects_set) == num_distractors + 2
                assert left in grid.objects_set
                assert right in grid.objects_set
                left_pos = grid.get_pos_for_object(left)
                right_pos = grid.get_pos_for_object(right)
                if isinstance(prep, relations.LeftOf):
                    assert left_pos[1] < right_pos[1]
                else:
                    assert left_pos[0] < right_pos[0]
            print('')
            print('neg examples:')
            failed_conditions_count = Counter()
            for i in range(100):
                grid = relation.create_negative_example(num_distractors=num_distractors, grid_size=5)
                if i < 3 or True:
                    print(grid)
                    print('')
                assert len(grid.objects_set) == num_distractors + 2
                failed_conditions = []  # we want at least one failed, since negative example
                if left not in grid.objects_set:
                    failed_conditions.append('left missing')
                if right not in grid.objects_set:
                    failed_conditions.append('right missing')
                if len(failed_conditions) == 0:
                    left_pos = grid.get_pos_for_object(left)
                    right_pos = grid.get_pos_for_object(right)
                    if isinstance(prep, relations.LeftOf):
                        if left_pos[1] >= right_pos[1]:
                            failed_conditions.append('relation wrong')
                    else:
                        if left_pos[0] >= right_pos[0]:
                            failed_conditions.append('relation wrong')
                # print('failed_conditions', failed_conditions)
                assert len(failed_conditions) == 1
                failed_conditions_count[failed_conditions[0]] += 1
            print('failed_conditions_count', sorted(failed_conditions_count.items()))
            assert len(failed_conditions_count) == 3  # left mismatch, right mismatch, relations mismatch


def test_relation_hg_seed():
    r1 = np.random.RandomState(123)
    r2 = np.random.RandomState(123)
    num_holdout = 3
    generator1 = hg.RelationsHG(
        r=r1, available_preps=['LeftOf', 'Above'],
        num_avail_colors=9, num_avail_shapes=9, num_holdout=num_holdout)
    generator2 = hg.RelationsHG(
        r=r2, available_preps=['LeftOf', 'Above'],
        num_avail_colors=9, num_avail_shapes=9, num_holdout=num_holdout)
    assert generator1.thing_space_training.color_space.sample() == \
        generator2.thing_space_training.color_space.sample()
    assert generator1.thing_space_training.shape_space.sample() == \
        generator2.thing_space_training.shape_space.sample()
    assert generator1.thing_space_training.sample() == \
        generator2.thing_space_training.sample()
    for train in [True, False]:
        relation1 = generator1.sample_hyp(train=train)
        relation2 = generator2.sample_hyp(train=train)
        print('relation1', relation1)
        print('relation2', relation2)
        assert relation1 == relation2

        for num_distractors in range(3):
            for i in range(100):
                grid1 = relation1.create_positive_example(num_distractors=num_distractors, grid_size=5)
                grid2 = relation2.create_positive_example(num_distractors=num_distractors, grid_size=5)
                assert grid1 == grid2
            for i in range(100):
                grid1 = relation1.create_negative_example(num_distractors=num_distractors, grid_size=5)
                grid2 = relation2.create_negative_example(num_distractors=num_distractors, grid_size=5)
                assert grid1 == grid2


@pytest.mark.parametrize(
    "HGClass,num_colors,holdout_fraction,expected_holdout",
    [
        (hg.RelationsHG, 3, 0.2, 2),
        (hg.RelationsHG, 9, 0.2, 17),
        (hg.Colors1HG, 3, 0.2, 1),
        (hg.Colors1HG, 4, 0.2, 1),
        (hg.Colors1HG, 5, 0.2, 1),
        (hg.Colors1HG, 9, 0.2, 2),
        (hg.Colors2HG, 3, 0.2, -1),
        (hg.Colors2HG, 4, 0.2, 2),
        (hg.Colors2HG, 5, 0.2, 3),
        (hg.Colors2HG, 9, 0.2, 9),
        (hg.Colors3HG, 3, 0.2, -1),
        (hg.Colors3HG, 4, 0.2, -1),
        (hg.Colors3HG, 5, 0.2, 7),
        (hg.Colors3HG, 9, 0.2, 33),
        (hg.Shapes1HG, 3, 0.2, 1),
        (hg.Shapes1HG, 4, 0.2, 1),
        (hg.Shapes1HG, 5, 0.2, 1),
        (hg.Shapes1HG, 9, 0.2, 2),
        (hg.Shapes2HG, 3, 0.2, -1),
        (hg.Shapes2HG, 4, 0.2, 2),
        (hg.Shapes2HG, 5, 0.2, 3),
        (hg.Shapes2HG, 9, 0.2, 9),
        (hg.Shapes3HG, 3, 0.2, -1),
        (hg.Shapes3HG, 4, 0.2, -1),
        (hg.Shapes3HG, 5, 0.2, 7),
        (hg.Shapes3HG, 9, 0.2, 33),
        (hg.Things1HG, 3, 0.2, 2),
        (hg.Things2HG, 3, 0.2, 2),
        (hg.Things3HG, 3, 0.2, 3),
        (hg.Things3HG, 9, 0.2, 17),
    ]
)
def test_num_holdout(HGClass, num_colors: int, holdout_fraction: float, expected_holdout: int):
    """
    we set num_shapes = num_colors
    """
    num_distractors = 2
    num_holdout = HGClass.get_num_holdout(
        num_avail_colors=num_colors,
        num_distractors=num_distractors,
        num_avail_shapes=num_colors,
        holdout_fraction=holdout_fraction)
    print('num_holdout', num_holdout)
    assert num_holdout == expected_holdout
