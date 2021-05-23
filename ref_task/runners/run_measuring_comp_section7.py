"""
Run experiment from "measuring compositionality" section 7, using texrel dataset
We will run 'run_end_to_end.py' 100 times, using texrel, for ~400 steps each time,
and store the tre, rho, uniqueness, holdout acc, and training acc

Note to self: example dev run command
python ref_task/runners/run_measuring_comp_section7.py --ref foo --disable-cuda --ds-collection debugmac \
    --render-every-seconds 1 --max-steps 10 --batch-size 8 --num-runs 3 --tre-steps 10 --tre-max-samples 10
"""
import argparse
import csv
import os
import datetime
import sys
import mlflow

from ulfs import git_info
from ulfs.params import Params

from ref_task import params_groups, run_end_to_end


class FakeRunner(object):
    def __init__(self, parser):
        self.parser = parser

    def add_param(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)


def run(args):
    results = []
    for it in range(args.num_runs):
        child_params = Params(dict(args.__dict__))
        child_params.ref = f'{args.ref}_r{it:02}'
        # del child_params.__dict__['fast_dev_run']
        child_seed = args.seed_base + it
        child_params.seed = child_seed
        del child_params.__dict__['seed_base']
        child_params.evaluate_tre_on_terminate = True

        runner = run_end_to_end.Runner()
        runner._extract_standard_args(child_params)
        runner.enable_cuda = child_params.enable_cuda
        runner.setup_base(params=child_params)
        runner.run_base()
        res = runner.res
        print('res', res)

        res['name'] = f'r{it:02}'
        # what things do we need, conceptually? we want:
        # batch
        # elapsed_time
        # val_same_tre
        # val_same_rho
        # val_same_uniq
        # val_same_acc
        # val_new_acc
        name_mapper = {
            'batch': 'b',
            'elapsed_time': 't',
            # 'val_same_tre': 'tre',
            # 'val_same_rho': 'rho',
            # 'val_same_prec': 'prec',
            # 'val_same_rec': 'rec',
            # 'val_same_gnd_clusters': 'gnd_cls',
            # 'val_same_pred_clusters': 'pred_cls',
            # 'val_same_acc': 'same_acc',
            # 'val_new_acc': 'new_acc'
        }
        res = {name_mapper.get(k, k): v for k, v in res.items()}
        print('res', res)
        res['t'] = str(int(res['t']))
        res['seed'] = child_seed
        res_keys = [
            'name', 'seed', 'b', 't']
        for split_name in ['val_same', 'val_new', 'test_same', 'test_new']:
            for metric_name in ['acc', 'rho', 'prec', 'rec', 'gnd_clusters', 'pred_clusters', 'tre']:
                res_keys.append(f'{split_name}_{metric_name}')
            # res_keys += [
            #     f'{split_name}_acc', f'{split_name}_rho', f'{split_name}_prec', f'{split_name}_rec',
            #     f'{split_name}_gnd_clusters', f'{split_name}_pred_clusters', f'{split_name}_'
            # ]
            # 'tre', 'rho', 'prec', 'rec',
            # 'gnd_cls', 'pred_cls',
            # 'same_acc', 'new_acc']
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

    if 'MLFLOW_TRACKING_URI' in os.environ:
        mlflow.set_experiment('hp/run_obv')
        mlflow.start_run(run_name=args.ref + '_summary')

        with open(args.out_csv, 'r') as f_in:
            csv_text = f_in.read()
        mlflow.log_text(csv_text, f'{args.ref}.csv')

        meta = {}
        meta['params'] = args.__dict__
        meta['argv'] = sys.argv
        meta['hostname'] = os.uname().nodename
        meta['gitlog'] = git_info.get_git_log()
        meta['gitdiff'] = git_info.get_git_diff()
        meta['start_datetime'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        meta['argv'] = ' '.join(meta['argv'])
        gitdiff = meta['gitdiff']
        gitlog = meta['gitlog']
        mlflow.log_text(gitdiff, 'gitdiff.txt')
        mlflow.log_text(gitlog, 'gitlog.txt')

        params_to_log = dict(meta['params'])
        for too_long in []:
            _contents = params_to_log[too_long]
            if isinstance(_contents, list):
                _contents = '\n'.join(_contents)
            del params_to_log[too_long]
            mlflow.log_text(_contents, f'{too_long}.txt')

        mlflow.log_params(params_to_log)

        del meta['gitdiff']
        del meta['gitlog']
        del meta['params']
        mlflow.log_params(meta)
        mlflow.set_tags(meta)

        mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--out-csv-templ', type=str, default='{ref}.csv')
    parser.add_argument('--num-runs', type=int, default=100)
    parser.add_argument('--seed-base', type=int, default=123)
    # parser.add_argument('--fast-dev-run', action='store_true')

    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--save-every-seconds', type=int, default=-0)
    parser.add_argument('--render-every-seconds', type=int, default=30)
    parser.add_argument('--render-every-steps', type=int, default=-1)
    parser.add_argument('--name', type=str, default='run_end_to_end', help='used for logfile naming')
    parser.add_argument('--load-last-model', action='store_true')
    parser.add_argument('--model-file', type=str, default='tmp/{name}_{ref}_{hostname}_{date}_{time}.dat')
    parser.add_argument('--logfile', type=str, default='logs/log_{name}_{ref}_{hostname}_{date}_{time}.log')

    # we use the fake runner to add arguments to the parser object
    runner = FakeRunner(parser=parser)

    run_end_to_end.add_e2e_args(runner)

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
    run(args)
