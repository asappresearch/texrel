"""
create all data for specific collection

this will use ulfs_submit.py, use single job
per dataset
"""
import argparse
import time
import subprocess
import os

from texrel import create_collection


def run(args):
    sub_refs = []
    for hypothesis_generator in args.hypothesis_generators:
        for dist in args.num_distractors:
            num_holdout = create_collection.get_num_holdout(
                hypothesis_generator=hypothesis_generator,
                num_colors=args.num_colors,
                num_shapes=args.num_shapes,
                holdout_fraction=args.holdout_fraction,
                num_distractors=dist
            )
            if num_holdout < 0:
                continue
            sub_ref = f'{args.ref}-{hypothesis_generator.lower()}-d{dist}'
            sub_cmd = f'ulfs_submit.py -r {sub_ref} --gpus 0 --cpus 1 --memory 8 --no-follow --ref-arg ds-ref'
            sub_cmd += ' --'
            sub_cmd += ' texrel/create_dataset.py'
            for k in [
                    'seed',
                    'inner_train_pos', 'inner_train_neg', 'inner_test_pos', 'inner_test_neg',
                    'num_train', 'num_val_same', 'num_val_new', 'num_test_same', 'num_test_new',
                    'num_colors', 'num_shapes',
                    'available_preps'
            ]:
                arg_key = '--' + k.replace('_', '-')
                arg_value = getattr(args, k)
                sub_cmd += f' {arg_key} {arg_value}'
            sub_cmd += f' --num-holdout {num_holdout}'
            sub_cmd += f' --hypothesis-generator {hypothesis_generator}'
            sub_cmd += f' --num-distractors {dist}'
            print(sub_cmd)
            if not args.dry_run:
                os.system(sub_cmd)
            sub_refs.append(sub_ref)
    if not args.dry_run:
        while True:
            for sub_ref in sub_refs:
                print(sub_ref)
                logs = subprocess.check_output(['ulfs_logs.sh', sub_ref]).decode('utf-8').split('\n')[-5:]
                print('\n'.join(logs))
            time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-distractors', type=str, default='0,2', help='comma-separated')
    parser.add_argument(
        '--hypothesis-generators', type=str,
        default='Colors1,Colors2,Colors3,Shapes1,Shapes2,Shapes3,Things1,Things2,Things3,Relations',
        help='comma-separated')

    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dry-run', action='store_true')

    parser.add_argument('--num-colors', type=int, default=9)
    parser.add_argument('--num-shapes', type=int, default=9)
    parser.add_argument('--available-preps', type=str, default='LeftOf,Above', help='default is LeftOf,Above')
    # parser.add_argument('--num-holdout-base', type=int, default=4,
    # help='we will square this for xx2, and cube for xx3')
    parser.add_argument('--holdout-fraction', type=float, default=0.2,
                        help='we will round up the actual holdout count to nearest 1')

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
    args = parser.parse_args()
    args.num_distractors = [int(v) for v in args.num_distractors.split(',')]
    args.hypothesis_generators = args.hypothesis_generators.split(',')
    if args.fast_dev_run:
        args.num_train = 32
        args.num_val_same = 32
        args.num_val_new = 32
        args.num_test_same = 32
        args.num_test_new = 32
        args.num_distractors = [2]
        args.hypothesis_generators = ['Colors1', 'Things1']
    del args.__dict__['fast_dev_run']
    run(args)
