"""
create all data for specific collection
"""
import argparse
from os.path import expanduser as expand

from texrel import create_dataset, hypothesis_generators as hg


def get_num_holdout(
        hypothesis_generator: str, num_distractors: int,
        num_colors: int, num_shapes: int, holdout_fraction: float) -> int:
    """
    if returns -1, then configuration not valid
    """
    HGClass: hg.HypothesisGenerator2 = getattr(hg, f'{hypothesis_generator}HG')
    num_holdout = HGClass.get_num_holdout(
        num_avail_colors=num_colors, num_distractors=num_distractors,
        num_avail_shapes=num_shapes, holdout_fraction=holdout_fraction)
    print('HGClass', HGClass, 'holdout_number', num_holdout)
    return num_holdout


def run(args):
    for hypothesis_generator in args.hypothesis_generators:
        for dist in args.num_distractors:
            num_holdout = get_num_holdout(
                hypothesis_generator=hypothesis_generator,
                num_colors=args.num_colors,
                num_shapes=args.num_shapes,
                holdout_fraction=args.holdout_fraction,
                num_distractors=dist,
            )
            if num_holdout < 0:
                continue
            child_args = dict(args.__dict__)
            child_args['num_holdout'] = num_holdout
            # del child_args.__dict__['num_holodout_base']
            del child_args['holdout_fraction']
            child_args['num_distractors'] = dist
            child_args['hypothesis_generator'] = hypothesis_generator
            sub_ref = f'{args.ref}-{hypothesis_generator.lower()}-d{dist}'
            child_args['ds_ref'] = sub_ref
            child_args['out_filepath'] = args.out_filepath.format(ds_ref=child_args['ds_ref'])
            # del child_args['num_distractors']
            del child_args['ref']
            del child_args['hypothesis_generators']
            dataset_generator = create_dataset.DatasetGenerator(**child_args)
            dataset_generator.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-distractors', type=str, default='0,2', help='comma-separated')
    parser.add_argument(
        '--hypothesis-generators', type=str,
        default='Colors1,Colors2,Colors3,Shapes1,Shapes2,Shapes3,Things1,Things2,Things3,Relations',
        help='comma-separated')
    parser.add_argument('--available-preps', type=str, default='LeftOf,Above', help='default is LeftOf,Above')

    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--num-colors', type=int, default=9)
    parser.add_argument('--num-shapes', type=int, default=9)
    # parser.add_argument('--num-holdout-base', type=int, default=4)
    parser.add_argument('--holdout-fraction', type=float, default=0.2,
                        help='we will round up the actual holdout count to nearest 1')
    parser.add_argument('--inner-train-pos', type=int, default=3, help='inner train pos')
    parser.add_argument('--inner-train-neg', type=int, default=3, help='inner train neg')
    parser.add_argument('--inner-test-pos', type=int, default=1, help='inner test pos')
    parser.add_argument('--inner-test-neg', type=int, default=0, help='inner test neg')
    parser.add_argument('--out-filepath', type=str, default=expand('~/data/reftask/{ds_ref}.dat'))
    parser.add_argument('--num-train', type=int, default=100000, help='num outer train')
    parser.add_argument('--num-val-same', type=int, default=1024, help='num outer val_same')
    parser.add_argument('--num-val-new', type=int, default=1024, help='num outer val_new')
    parser.add_argument('--num-test-same', type=int, default=1024, help='num outer test_same')
    parser.add_argument('--num-test-new', type=int, default=1024, help='num outer test_new')
    parser.add_argument('--fast-dev-run', action='store_true', help='set nums to 128 or so')
    args = parser.parse_args()
    args.num_distractors = [int(v) for v in args.num_distractors.split(',')]
    args.hypothesis_generators = args.hypothesis_generators.split(',')
    args.available_preps = args.available_preps.split(',')
    if args.fast_dev_run:
        args.num_train = 32
        args.num_val_same = 32
        args.num_val_new = 32
        args.num_test_same = 32
        args.num_test_new = 32
        args.num_distractors = [0, 2]
        # args.hypothesis_generators = ['Colors1', 'Things1']
    del args.__dict__['fast_dev_run']
    run(args)
