import time

import pytest
from colorama import init as colorama_init

from ulfs import ascii_render

from texrel import dataset
from texrel.dataset import Dataset
from texrel import relations, spaces


@pytest.mark.skip
def test_eyeballing():
    colorama_init()

    relation_space = relations.RelationSpace()
    dataset = Dataset(grid_size=5, rel_space=relation_space)
    batch = dataset.sample(
        batch_size=5, num_pos=0, num_neg=0, num_distractors=2)
    ex_l_2 = dataset.tensor_to_grid_l(
        tensor=batch['receiver_examples_t'])
    for i, r in enumerate(batch['relations_l']):
        print(r)
        print('comp', r.complement())
        if batch['labels_t'][i].item() == 1:
            print('Pos')
        else:
            print('Neg')
        print('receiver_example')
        print(batch['receiver_examples_l'][i])
        print('')
        print(ex_l_2[i])
        print('')

    # r = generate_relation()
    # print(r)
    # print(r.complement())
    # grid = r.create_example_gridform(grid_size=5, num_distractors=2)
    # print(grid)
    # relations_l = generate_relations_stack(batch_size=5)
    # # print('relations_l', relations_l[0])
    # examples_t = generate_positive_examples(relations_l)

    # batch = create_batch(batch_size=3)
    # print('r', r)
    # print('r.encode_onehot()', r.encode_onehot())
    # print('batch', batch)

    # print('examples_t', examples_t[0])


@pytest.mark.skip
def test_rec_gridsize():
    grid_size = 2
    relation_space = relations.RelationSpace()
    d = Dataset(grid_size=grid_size, rel_space=relation_space)
    s = d.sample(batch_size=2, num_pos=0, num_neg=0, num_distractors=0)
    print('s.keys()', s.keys())
    rec_ex = s['receiver_examples_t']
    print('rec_ex.size()', rec_ex.size())
    assert len(rec_ex.size()) == 4
    assert rec_ex.size()[2] == grid_size
    assert rec_ex.size()[3] == grid_size

    grid_size = 5
    relation_space = relations.RelationSpace()
    d = Dataset(grid_size=grid_size, rel_space=relation_space)
    s = d.sample(batch_size=2, num_pos=0, num_neg=0, num_distractors=0)
    print('s.keys()', s.keys())
    rec_ex = s['receiver_examples_t']
    assert len(rec_ex.size()) == 4
    assert rec_ex.size()[2] == grid_size
    assert rec_ex.size()[3] == grid_size


@pytest.mark.skip
def test_num_distractors():
    grid_size = 5
    for num_distractors in range(3):
        relation_space = relations.RelationSpace()
        d = Dataset(grid_size=grid_size, rel_space=relation_space)
        s = d.sample(batch_size=2, num_pos=0, num_neg=0, num_distractors=num_distractors)
        print('s.keys()', s.keys())
        # # rec_ex = s['receiver_examples_l']
        # grid_ex = rec_ex[0].grid
        # num_objects = 0
        # for i in range(grid_size):
        #     for j in range(grid_size):
        #         if grid_ex[i][j] is not None:
        #             num_objects += 1
        # print('num_objects', num_objects)
        # assert num_objects == 2 + num_distractors


def get_coord(o, grid):
    grid_size = len(grid)
    for h in range(grid_size):
        for w in range(grid_size):
            if grid[h][w] is not None and grid[h][w] == o:
                return [h, w]


@pytest.mark.skip
def test_input_samples():
    """
    we want to make sure that each not- instance generates at least one instance
    where the objects are aligned, and one where they are not aligned
    since it's hard to test this without white-boxing, we're going to take samples,
    and check any not- instances; and for each one count the number of aligned
    and non-aligned instances; and check ok
    """
    print('')
    relation_space = relations.RelationSpace()
    grid_size = 5
    num_pos = 5
    num_neg = 0
    # num_input_examples = 5
    d = Dataset(grid_size=grid_size, rel_space=relation_space)
    num_nots_seen = 0
    while num_nots_seen < 1000:
        # set distractors to zero to facilitate our testing/analysis
        # adding distractors is easy to test separately anyway
        # (given we know how the implementation works...)
        s = d.sample(batch_size=1, num_pos=num_pos, num_neg=num_neg, num_distractors=0)
        r = s['relations_l'][0]
        prep_class = r.prep.__class__
        if prep_class not in [
                relations.NotRightOf, relations.NotLeftOf, relations.NotAbove, relations.NotBelow]:
            continue
        num_nots_seen += 1
        # input_examples_l = s['input_examples_l']
        # input_examples = [b[0] for b in input_examples_l]

        # num_aligned = 0
        # num_not_aligned = 0
        # for n in range(num_input_examples):
        #     coords = []
        #     grid = input_examples[n].grid
        #     left_coord = get_coord(r.left, grid)
        #     right_coord = get_coord(r.right, grid)

        #     align_coords = []  # we'll only consider the coords in the axis for the preposition
        #                        # ie width for left/right, height for above/below
        #     if prep_class in [relations.NotRightOf, relations.NotLeftOf]:
        #         align_coords.append(left_coord[1])
        #         align_coords.append(right_coord[1])
        #     elif prep_class in [relations.NotAbove, relations.NotBelow]:
        #         align_coords.append(left_coord[0])
        #         align_coords.append(right_coord[0])
        #     else:
        #         raise Exception('shouldnt be here')
        #     if align_coords[0] == align_coords[1]:
        #         num_aligned += 1
        #     else:
        #         num_not_aligned += 1
        # assert num_aligned > 0
        # assert num_not_aligned > 0


@pytest.mark.skip
def test_relations_t():
    print('')
    relation_space = relations.RelationSpace()
    grid_size = 5
    # num_input_examples = 5
    num_pos = 3
    num_neg = 3
    batch_size = 8
    print('batch_size', batch_size)
    # print('num_input_examples', num_input_examples)
    d = Dataset(grid_size=grid_size, rel_space=relation_space)
    s = d.sample(batch_size=batch_size, num_pos=num_pos, num_neg=num_neg, num_distractors=0)
    relations_t = s['relations_t']
    print('relations_t', relations_t)
    print('relations_t.size()', relations_t.size())


@pytest.mark.skip
def test_time_sampling():
    """
    how many batches can we generate a second?
    """
    grid_size = 5
    num_pos = 3
    num_neg = 3
    # num_input_examples = 5
    batch_size = 128
    num_distractors = 2

    relation_space = relations.RelationSpace()

    print('batch_size', batch_size)
    # print('num_input_examples', num_input_examples)
    d = Dataset(grid_size=grid_size, rel_space=relation_space)
    start_time = time.time()
    num_batches = 0
    while time.time() - start_time < 3.0:
        d.sample(batch_size=batch_size, num_pos=num_pos, num_neg=num_neg, num_distractors=num_distractors)
        num_batches += 1
    print('batches per second %.1f' % (num_batches / (time.time() - start_time)))


@pytest.mark.skip
def test_time_holdout_sampling():
    """
    how many batches can we generate a second?
    """
    num_holdout = 5
    grid_size = 5
    num_pos = 3
    num_neg = 3
    # num_input_examples = 5
    batch_size = 128
    num_distractors = 2

    thing_space = spaces.ThingSpace()
    train_space, test_space = thing_space.partition(
        [thing_space.num_unique_things - num_holdout, num_holdout])
    thing_space = test_space

    relation_space = relations.RelationSpace(thing_space=thing_space)

    print('batch_size', batch_size)
    # print('num_input_examples', num_input_examples)
    d = Dataset(grid_size=grid_size, rel_space=relation_space)
    start_time = time.time()
    num_batches = 0
    while time.time() - start_time < 3.0:
        d.sample(batch_size=batch_size, num_pos=num_pos, num_neg=num_neg, num_distractors=num_distractors)
        num_batches += 1
    print('batches per second %.1f' % (num_batches / (time.time() - start_time)))


@pytest.mark.skip
def test_create_input_labels():
    batch_size = 16
    num_pos = 3
    num_neg = 3
    input_labels = dataset.create_input_labels(
        batch_size=batch_size,
        num_pos=num_pos,
        num_neg=num_neg
    )
    print('input_labels', input_labels)


@pytest.mark.skip
def test_input_examples():
    batch_size = 4
    num_pos = 3
    num_neg = 3
    grid_size = 5

    relation_space = relations.RelationSpace()
    d = Dataset(grid_size=grid_size, rel_space=relation_space)
    s = d.sample(batch_size=batch_size, num_pos=num_pos, num_neg=num_neg, num_distractors=0)
    print('s.keys()', s.keys())
    input_labels = s['input_labels_t']
    inputs = s['input_examples_t']
    rs = s['relations_l']
    print('inputs.size()', inputs.size())
    screen = ascii_render.AsciiScreen(1 + 8 * batch_size, 118)
    h = 1
    num_examples = inputs.size(0)
    print('num_examples', num_examples)
    for n in range(batch_size):
        screen.print(h, 1, rs[n])
        h += 1
        w = 1
        for j in range(num_examples):
            label = input_labels[j, n].item()
            # assert inputs[j, n, 0].min() == label
            # assert inputs[j, n, 0].max() == label
            screen.print(h, w, '%s' % int(label))
            grid = inputs[j, n]
            screen.print(h + 1, w, d.tensor_to_grid(grid))
            w += 8
        h += 7
    screen.render()
