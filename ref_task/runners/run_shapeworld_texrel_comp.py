"""
compare texrel and shapeworld datasets

this also runs shapeworld using lsl code
expects to find lsl repo in same folder that this repo is inside, i.e this repo
and lsl repo should be peers, inside the same folder.
"""
import argparse
import csv
import time
import sys
import subprocess
from typing import List, Dict, Any
from os.path import expanduser as expand

from ulfs.params import Params
from ref_task import params_groups, run_end_to_end


class FakeRunner(object):
    def __init__(self, parser):
        self.parser = parser

    def add_param(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)


def lsl_rows_to_metrics(lsl_rows: List[str]) -> Dict[str, Any]:
    """
    take rows similar to below, and return metrics and epoch number
====>      (train)	Epoch:  50	Loss: 69.3388
mean accuracy during training 0.508
====>      (train)	Epoch:  50	Accuracy: 0.5063
====>        (val)	Epoch:  50	Accuracy: 0.4920
====>       (test)	Epoch:  50	Accuracy: 0.5040
====>   (val_same)	Epoch:  50	Accuracy: 0.5040
====>  (test_same)	Epoch:  50	Accuracy: 0.5032
====> DONE
====> BEST EPOCH: 3
====>        (best_val)	Epoch: 3	Accuracy: 0.5080
====>   (best_val_same)	Epoch: 3	Accuracy: 0.5200
====>       (best_test)	Epoch: 3	Accuracy: 0.5060
====>  (best_test_same)	Epoch: 3	Accuracy: 0.5048
====>
====>    (best_val_avg)	Epoch: 3	Accuracy: 0.5140
====>   (best_test_avg)	Epoch: 3	Accuracy: 0.5054
"""
    metrics = {}
    last_epoch = 0
    train_acc_by_epoch = {}
    for row in lsl_rows:
        if '(best_' in row:
            metric_name = row.split('(')[1].split(')')[0]
            epoch = int(row.split('Epoch: ')[1].split()[0])
            value = float(row.split('Accuracy: ')[1].split()[0])
            metrics[metric_name] = value
            metrics['best_epoch'] = epoch

        if 'Epoch: ' in row:
            last_epoch = max(last_epoch, int(row.split('Epoch:')[1].lstrip().split()[0]))

        if '(train)' in row:
            _epoch = int(row.split('Epoch:')[1].strip().split()[0])
            _acc = float(row.strip().split()[-1])
            train_acc_by_epoch[_epoch] = _acc
    metrics['total_epochs'] = last_epoch
    print('train_acc_by_epoch', train_acc_by_epoch)
    metrics['train_acc'] = train_acc_by_epoch[int(metrics['best_epoch'])]
    return metrics


def run_lsl_shapeworld(
        seed: int, data_dir: str, epochs: int, enable_cuda: bool, soft_test: bool, batch_size: int,
        augment: bool):
    extra_args = []
    if enable_cuda:
        extra_args += ['--cuda']
    if soft_test:
        extra_args += ['--soft_test']
    if not augment:
        extra_args += ['--no_augment']
    start_time = time.time()
    lsl_results = subprocess.check_output([
        sys.executable,
        'lsl/train.py', '--infer_hyp', '--hypo_lambda', '1.0', '--batch_size', str(batch_size),
        '--seed', str(seed), '--e2e_emergent_communications',
        '--data_dir', data_dir, 'exp/l3', '--backbone', 'conv4',
        '--epochs', str(epochs)
    ] + extra_args, cwd='../lsl/shapeworld').decode('utf-8')
    elapsed = time.time() - start_time
    rows = lsl_results.split('\n')
    metrics = lsl_rows_to_metrics(rows)
    print('metrics', metrics)
    metrics['total_time'] = elapsed
    metrics['time_till_best'] = metrics['best_epoch'] / metrics['total_epochs'] * elapsed
    return metrics


def run(args):
    results = []
    res_keys = [
        'ds_family', 'sampler_model', 'augment',
        'seed', 'b', 't',
        'terminate_reason',
        'train_acc']
    #     'val_same_acc', 'val_same_rho', 'val_same_prec', 'val_same_rec',
    #     'val_same_gnd_clusters', 'val_same_pred_clusters',
    #     'val_new_acc', 'val_new_rho', 'val_new_prec', 'val_new_rec',
    #     'val_new_gnd_clusters', 'val_new_pred_clusters',
    #     'test_same_acc', 'test_new_acc'
    # ]
    for split_name in ['val_same', 'val_new', 'test_same', 'test_new']:
        for metric_name in ['acc', 'rho', 'prec', 'rec', 'gnd_clusters', 'pred_clusters']:
            res_keys.append(f'{split_name}_{metric_name}')

    print(args)

    def write_results():
        with open(args.out_csv, 'w') as f_out:
            dict_writer = csv.DictWriter(f_out, fieldnames=res_keys)
            dict_writer.writeheader()
            for row in results:
                dict_writer.writerow(row)
                print(row)
        print('wrote', args.out_csv)

    for augment in [False, True]:
        for soft_test in [True, False]:
            _lsl_metrics = run_lsl_shapeworld(
                seed=args.seed_base,
                data_dir=args.lsl_data_dir,
                epochs=args.lsl_epochs,
                enable_cuda=not args.disable_cuda,
                soft_test=soft_test,
                batch_size=args.lsl_batch_size,
                augment=augment,
            )
            res = {
                'ds_family': 'lsl_shapeworld',
                'sampler_model': 'soft' if soft_test else 'discrete',
                'augment': augment,
                'seed': args.seed_base,
                # 't_total': str(int(_lsl_metrics['total_time'])),
                't': str(int(_lsl_metrics['time_till_best'])),
                'train_acc': '%.3f' % _lsl_metrics['train_acc'],
                'val_same_acc': '%.3f' % _lsl_metrics['best_val_same'],
                'val_new_acc': '%.3f' % _lsl_metrics['best_val'],
                'test_same_acc': '%.3f' % _lsl_metrics['best_test_same'],
                'test_new_acc': '%.3f' % _lsl_metrics['best_test'],
            }
            print(res)
            results.append(res)
            write_results()

    for ds_family in args.ds_families:
        for sampler_model in args.sampler_models:
            print('')
            print('=======================================')
            print(ds_family, sampler_model)
            child_params = Params(dict(args.__dict__))
            child_params.ref = f'{args.ref}_{ds_family}_{sampler_model}'
            child_params.ds_tasks = args.hg
            child_params.ds_family = ds_family
            child_params.sampler_model = sampler_model
            child_seed = args.seed_base + 0
            child_params.seed = child_seed
            del child_params.__dict__['seed_base']

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
            res['ds_family'] = ds_family
            res['sampler_model'] = sampler_model
            res['augment'] = False
            print('res', res)
            res['t'] = str(int(res['t']))
            res['seed'] = child_seed
            res = {k: res[k] for k in res_keys}
            for k in res_keys:
                if isinstance(res[k], float):
                    res[k] = '%.3f' % res[k]
            results.append(res)
            write_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lsl-batch-size', type=int, default=100)
    # parser.add_argument('--lsl-soft-test', action='store_true')
    parser.add_argument('--lsl-epochs', type=int, default=50)
    parser.add_argument('--lsl-data-dir', type=str, default=expand('~/data'))

    parser.add_argument('--send-arch', type=str, default='PrototypicalSender')
    parser.add_argument('--recv-arch', type=str, default='PrototypicalReceiver')

    parser.add_argument('--sampler-models', type=str, default='Softmax,Gumbel')
    parser.add_argument('--ds-families', type=str, default='shapeworld,texrel')

    parser.add_argument('--hg', type=str, default='Relations')

    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--out-csv-templ', type=str, default='{ref}.csv')
    parser.add_argument('--seed-base', type=int, default=123)
    parser.add_argument('--max-mins', type=float, help='finish running if reach this elapsed time')

    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='how many validations to run before early stopping')
    parser.add_argument('--early-stop-metric', type=str)
    parser.add_argument('--early-stop-direction', type=str, default='max')

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
    parser.add_argument('--render-every-seconds', type=int, default=-1)
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
    args.ds_families = args.ds_families.split(',')
    args.sampler_models = args.sampler_models.split(',')
    run(args)
