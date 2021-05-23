"""
Uses the new generators, which contain the relevant spaces, for both training and holdout,
rather than injecting the spaces from outside.
"""
import argparse
from typing import Tuple, List, Type
from os.path import expanduser as expand
import time

import torch
import numpy as np

from ulfs import file_utils
from texrel import hypothesis_generators as hgs, hypothesis


def create_examples_for_hypotheses(
        r: np.random.RandomState,
        num_pos: int, num_neg: int, grid_size: int, N: int, hypotheses_l: List[hypothesis.Hypothesis],
        num_distractors: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    given a list of hypotheses, generate examples for each hypothesis, and return
    them
    """
    labels = torch.zeros(num_pos + num_neg, N, dtype=torch.uint8)
    for n in range(N):
        labels_idx = torch.from_numpy(r.choice(num_pos + num_neg, num_pos, replace=False))
        labels[labels_idx, n] = 1

    shapes = torch.zeros(num_pos + num_neg, N, grid_size, grid_size, dtype=torch.uint8)
    colors = torch.zeros(num_pos + num_neg, N, grid_size, grid_size, dtype=torch.uint8)

    M = num_pos + num_neg

    last_print = time.time()
    print('generating examples ... ')
    for n in range(N):
        h = hypotheses_l[n]
        for m in range(M):
            label: int = labels[m][n].item()  # type: ignore
            grid = h.create_example(
                label=label, grid_size=grid_size, num_distractors=num_distractors)
            if n < 3 and m < 6:
                print('n', n, 'm', m, 'label', label)
                print(grid)
            input_shape, input_color = grid.as_shape_color_tensors()
            shapes[m, n] = input_shape
            colors[m, n] = input_color

        if time.time() - last_print >= 30.0:
            print('    n=', n)
            last_print = time.time()
    print('... generated ', N, 'examples')

    return labels, shapes, colors


def name_to_hash(name: str) -> int:
    """
    given a name, generate a unique-ish number.

    cannot simply use hash(), since that is different each
    time we re-run the process...
    """
    hash_v = sum([ord(c) for c in name])
    print('hash_v', hash_v)
    return hash_v


class DatasetGenerator(object):
    def __init__(
        self, hypothesis_generator, ds_ref,
        seed, out_filepath,
        grid_size, num_distractors,
        num_holdout,
        num_colors, num_shapes,
        inner_train_pos, inner_train_neg,
        inner_test_pos, inner_test_neg,
        available_preps: List[str],
        num_train, num_val_same, num_val_new,
        num_test_same, num_test_new,
        # in_ts_filepath,
    ):
        self.ds_ref = ds_ref
        self.seed = seed
        self.out_filepath = out_filepath
        self.available_preps = available_preps
        self.hypothesis_generator = hypothesis_generator
        self.num_colors = num_colors
        self.num_shapes = num_shapes
        self.inner_train_pos = inner_train_pos
        self.inner_train_neg = inner_train_neg
        self.inner_test_pos = inner_test_pos
        self.inner_test_neg = inner_test_neg
        self.grid_size = grid_size
        self.num_holdout = num_holdout
        self.num_distractors = num_distractors
        self.N_by_name = {
            'train': num_train,
            'val_same': num_val_same,
            'val_new': num_val_new,
            'test_same': num_test_same,
            'test_new': num_test_new
        }
        HypGen: Type[hgs.HypothesisGenerator2] = getattr(hgs, f'{hypothesis_generator}HG')
        print(HypGen.__name__)
        seed_offset = name_to_hash(HypGen.__name__) % 100  # type: ignore
        print('seed_offset', seed_offset)
        kwargs = {
            'num_holdout': num_holdout,
            'num_avail_colors': num_colors,
            'num_avail_shapes': num_shapes,
            'r': np.random.RandomState(seed=seed + seed_offset),
        }
        if HypGen == hgs.RelationsHG:
            kwargs['available_preps'] = available_preps
        self.hg: hgs.HypothesisGenerator2 = HypGen(  # type: ignore
            **kwargs
        )
        print('hg', self.hg)

    def generate_dataset(self, r: np.random.RandomState, N: int, train: bool):
        print('generating hypotheses...')
        hypotheses_l = [self.hg.sample_hyp(train=train) for n in range(N)]
        print('generated hypotheses')
        for h in hypotheses_l[:5]:
            print(h)
        hypotheses_english = [h.as_english() for h in hypotheses_l]
        hypotheses_structured = [h.as_english_structure() for h in hypotheses_l]
        words_set = set()
        max_seq_len = 0
        for i, h in enumerate(hypotheses_english):
            split_h = h.split()
            max_seq_len = max(max_seq_len, len(split_h))
            for word in split_h:
                words_set.add(word)
            if i <= 5:
                print('h', h)
        words = sorted(list(words_set))
        print('words', words)
        print('ground')
        _, ground_shapes, ground_colors = create_examples_for_hypotheses(
            r=r,
            hypotheses_l=hypotheses_l,
            num_pos=1,
            num_neg=0,
            grid_size=self.grid_size,
            N=N,
            num_distractors=0)
        print('inner_train')
        inner_train_labels, inner_train_shapes, inner_train_colors = create_examples_for_hypotheses(
            r=r,
            hypotheses_l=hypotheses_l,
            num_pos=self.inner_train_pos,
            num_neg=self.inner_train_neg,
            grid_size=self.grid_size,
            N=N,
            num_distractors=self.num_distractors)
        print('inner_test')
        inner_test_labels, inner_test_shapes, inner_test_colors = create_examples_for_hypotheses(
            r=r,
            hypotheses_l=hypotheses_l,
            num_pos=self.inner_test_pos,
            num_neg=self.inner_test_neg,
            grid_size=self.grid_size,
            N=N,
            num_distractors=self.num_distractors)
        return {
            'N': N,
            'words': words,
            'hypothesis_english_max_len': max_seq_len,
            'hypotheses_english': hypotheses_english,
            'hypotheses_structured': hypotheses_structured,
            'ground_shapes': ground_shapes.numpy(),
            'ground_colors': ground_colors.numpy(),
            'inner_train_labels': inner_train_labels.numpy(),
            'inner_train_shapes': inner_train_shapes.numpy(),
            'inner_train_colors': inner_train_colors.numpy(),
            'inner_test_shapes': inner_test_shapes.numpy(),
            'inner_test_colors': inner_test_colors.numpy(),
            'inner_test_labels': inner_test_labels.numpy()
        }

    def generate(self):
        dataset_by_name = {}
        r = np.random.RandomState(self.seed)
        for name, train in [
            ('train', True),
            ('val_same', True),
            ('val_new', False),
            ('test_same', True),
            ('test_new', False),
        ]:
            N = self.N_by_name[name]
            print('')
            print('=================================')
            print('split', name)
            dataset_by_name[name] = self.generate_dataset(
                r=r, N=N, train=train)
        save_dict = {
            'meta': {
                'ds_ref': self.ds_ref,
                'hypothesis_generators': [self.hypothesis_generator],
                'num_distractors': self.num_distractors,
                'grid_size': self.grid_size,
                'num_colors': self.num_colors,
                'num_shapes': self.num_shapes,
                'available_preps': self.available_preps,
                'num_holdout': self.num_holdout,
                'inner_train_pos': self.inner_train_pos,
                'inner_train_neg': self.inner_train_neg,
                'inner_test_pos': self.inner_test_pos,
                'inner_test_neg': self.inner_test_neg,
                # 'vocab_size': self.vocab_size,
                'seed': self.seed,
                'version': 'v3'
            },
            'data': dataset_by_name
        }
        file_utils.safe_save_pickled_gzip(
            self.out_filepath, save_dict)
        print('saved to ', self.out_filepath)


def run(**kwargs):
    generator = DatasetGenerator(**kwargs)
    generator.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--ds-ref', type=str, required=True)

    parser.add_argument('--hypothesis-generator', type=str, default='Colors1', help='just one generator')
    parser.add_argument('--num-colors', type=int, default=9)
    parser.add_argument('--num-shapes', type=int, default=9)
    parser.add_argument('--available-preps', type=str, default='LeftOf,Above', help='default is LeftOf,Above')
    parser.add_argument('--num-distractors', type=int, default=2)
    parser.add_argument('--num-holdout', type=int, default=4)
    parser.add_argument('--grid-size', type=int, default=5)

    parser.add_argument('--inner-train-pos', type=int, default=3, help='inner train pos')
    parser.add_argument('--inner-train-neg', type=int, default=3, help='inner train neg')
    parser.add_argument('--inner-test-pos', type=int, default=1, help='inner test pos')
    parser.add_argument('--inner-test-neg', type=int, default=0, help='inner test neg')

    parser.add_argument('--num-train', type=int, default=100000, help='num outer train')
    parser.add_argument('--num-val-same', type=int, default=1024, help='num outer val_same')
    parser.add_argument('--num-val-new', type=int, default=1024, help='num outer val_new')
    parser.add_argument('--num-test-same', type=int, default=1024, help='num outer test_same')
    parser.add_argument('--num-test-new', type=int, default=1024, help='num outer test_new')

    parser.add_argument('--fast-dev-run', action='store_true', help='set nums to 128 or so')

    parser.add_argument('--out-filepath', type=str, default=expand('~/data/reftask/{ds_ref}.dat'))

    args = parser.parse_args()
    args.out_filepath = args.out_filepath.format(**args.__dict__)
    args.available_preps = args.available_preps.split(',')
    if args.fast_dev_run:
        args.num_train = 128
        args.num_val_same = 128
        args.num_val_new = 128
        args.num_test_same = 128
        args.num_test_new = 128
    del args.__dict__['fast_dev_run']
    run(**args.__dict__)
