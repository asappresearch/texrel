"""
run for various combinations of datasets, to examine the effect
of multi-learning.
- Does learning Colors1 help Colors3
- Does learning Shapes1 help Colors3?
- Does learning Shapes1 or Colors1 help Things1?
- Does learning Things1 help Relations?
- Does learning all of them together help any of them?
"""
import argparse
import csv

from ulfs.params import Params
from ref_task import params_groups, run_end_to_end


class FakeRunner(object):
    def __init__(self, parser):
        self.parser = parser

    def add_param(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)


scenarios_by_name = {
    'none': {'dists': [2], 'sets': []},
    'colors12': {'dists': [2], 'sets': ['Colors1', 'Colors2']},
    'shapes12': {'dists': [2], 'sets': ['Shapes1', 'Shapes2']},
    'things12': {'dists': [2], 'sets': ['Things1', 'Things2']},
    'all': {
        'dists': [2], 'sets':
        'Colors1,Colors2,Colors3,Shapes1,Shapes2,Shapes3,Things1,Things2,Things3,Relations'.split(',')},
    'allalldists': {
        'dists': [0, 2], 'sets':
        'Colors1,Colors2,Colors3,Shapes1,Shapes2,Shapes3,Things1,Things2,Things3,Relations'.split(',')},
}


def run(args):
    results = []
    send_arch, recv_arch = args.arch_pair.split(':')
    for hg in args.left_hgs:
        for scenario_name in args.scenarios:
            scenario = scenarios_by_name[scenario_name]
            dists = scenario['dists']
            right_hgs = scenario['sets']
            all_hgs = set(right_hgs) | set([hg])

            child_params = Params(dict(args.__dict__))
            child_params.ref = f'{args.ref}_{send_arch}_{recv_arch}_{hg}_{scenario_name}'
            child_params.ds_tasks = ','.join(all_hgs)
            print('child_params.ds_tasks', child_params.ds_tasks)
            child_params.ds_distractors = ','.join([str(v) for v in dists])
            child_seed = args.seed_base + 0
            child_params.seed = child_seed
            del child_params.__dict__['seed_base']

            if args.arch_pair == 'RCNN:LearnedCNNMapping':
                child_params.batch_size //= 8

            print('child_params.batch_size', child_params.batch_size)

            runner = run_end_to_end.Runner()
            runner._extract_standard_args(child_params)
            runner.enable_cuda = child_params.enable_cuda
            runner.setup_base(params=child_params)
            runner.run_base()
            res = runner.res
            print('res', res)

            name_mapper = {
                'batch': 'b',
                'elapsed_time': 't',
                'acc': 'train_acc',
            }
            res = {name_mapper.get(k, k): v for k, v in res.items()}
            res['send_arch'] = send_arch
            res['recv_arch'] = recv_arch
            res['scenario'] = scenario_name
            res['bsz'] = child_params.batch_size
            res['hg'] = hg
            print('res', res)
            res['t'] = str(int(res['t']))
            res['seed'] = child_seed
            res_keys = [
                'send_arch', 'recv_arch', 'scenario', 'hg',
                'seed', 'b', 'bsz', 't',
                'terminate_reason',
                'train_acc'
            ]
            for split_name in ['val_same', 'val_new', 'test_same', 'test_new']:
                for metric_name in ['acc', 'rho', 'prec', 'rec', 'gnd_clusters', 'pred_clusters']:
                    res_keys.append(f'{split_name}_{metric_name}')
            res = {k: res[k] for k in res_keys}
            for k in res_keys:
                if isinstance(res[k], float):
                    res[k] = '%.3f' % res[k]
            results.append(res)
            with open(args.out_csv, 'w') as f_out:
                dict_writer = csv.DictWriter(f_out, fieldnames=res_keys)
                dict_writer.writeheader()
                for row in results:
                    dict_writer.writerow(row)
                    print(row)
            print('wrote', args.out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--arch-pair', type=str,
        default='PrototypicalSender:Cosine'
    )
    parser.add_argument(
        '--left-hgs', type=str,
        default='Colors3,Shapes3,Things3,Relations')
    parser.add_argument(
        '--scenarios', type=str,
        default='none,colors12,shapes12,things12,all,allalldists')

    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--out-csv-templ', type=str, default='{ref}.csv')
    parser.add_argument('--seed-base', type=int, default=123)
    # parser.add_argument('--tensor-dump-templ', type=str, default='tmp/{sub_ref}_{split_name}.pt')
    parser.add_argument('--max-mins', type=float, default=5, help='finish running if reach this elapsed time')
    parser.add_argument('--fast-dev-run', action='store_true')

    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='how many validations to run before early stopping')
    parser.add_argument('--early-stop-metric', type=str)
    parser.add_argument('--early-stop-direction', type=str, default='max')

    parser.add_argument('--sampler-model', type=str, default='Softmax')
    parser.add_argument('--no-sampler-hard', action='store_true')
    parser.add_argument('--sampler-tau', type=float, default=1.2)
    parser.add_argument('--sampler-gaussian-noise', type=float, default=0.0)
    parser.add_argument('--e2e-opt', type=str, default='Adam')
    parser.add_argument('--tensor-dumps-templ-path', type=str,
                        help='if not provided, then no dumps, use {batch} to represent batch number'
                        ' and {split_name} for split name')
    parser.add_argument('--tensor-dumps-templ-path-on-terminate', type=str,
                        help='if not provided, then no dumps, use'
                        ' {split_name} for split name')
    parser.add_argument('--evaluate-tre', action='store_true')
    parser.add_argument('--evaluate-tre-on-terminate', action='store_true', help='only when finish is True')

    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--save-every-seconds', type=int, default=-0)
    parser.add_argument('--render-every-seconds', type=int, default=30)
    parser.add_argument('--render-every-steps', type=int, default=300)
    parser.add_argument('--name', type=str, default='run_arch_send_comparison', help='used for logfile naming')
    parser.add_argument('--load-last-model', action='store_true')
    parser.add_argument('--model-file', type=str, default='tmp/{name}_{ref}_{hostname}_{date}_{time}.dat')
    parser.add_argument('--logfile', type=str, default='logs/log_{name}_{ref}_{hostname}_{date}_{time}.log')

    # we use the fake runner to add arguments to the parser object
    runner = FakeRunner(parser=parser)

    params_groups.add_ds_args(runner)  # type: ignore
    params_groups.add_e2e_args(runner)  # type: ignore
    params_groups.add_tre_args(runner)  # type: ignore
    params_groups.add_conv_args(runner)  # type: ignore
    params_groups.add_common_args(runner)  # type: ignore
    params_groups.add_sender_args(runner)  # type: ignore
    params_groups.add_receiver_args(runner)  # type: ignore

    args = parser.parse_args()
    args.out_csv = args.out_csv_templ.format(ref=args.ref)
    del args.__dict__['out_csv_templ']
    args.left_hgs = args.left_hgs.split(',')
    args.scenarios = args.scenarios.split(',')
    if args.fast_dev_run:
        args.render_every_seconds = 10
        args.max_mins = 0.05
        args.batch_size = 4
    run(args)
