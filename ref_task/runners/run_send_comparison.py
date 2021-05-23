"""
this file will only handle sender comparison

We'll run always with 2 distractors

We'll run:
- across each sender
- across each receiver
- for some end 2 end pairs
"""
import argparse
import csv

from ulfs.params import Params
from ref_task import params_groups, run_sender


class FakeRunner(object):
    def __init__(self, parser):
        self.parser = parser

    def add_param(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)


def run(args):
    results = []
    for send_architecture in args.sender_architectures:
        for hg in args.hgs:
            child_params = Params(dict(args.__dict__))
            child_params.ref = f'{args.ref}_{send_architecture}_{hg}'
            child_params.ds_tasks = hg
            child_seed = args.seed_base + 0
            child_params.seed = child_seed
            del child_params.__dict__['seed_base']
            child_params.image_seq_embedder = send_architecture
            if send_architecture in ['RCNN']:
                child_params.conv_preset = 'none'
            if send_architecture == 'RelationsTransformer':
                child_params.sender_decoder = 'IdentityDecoder'
                child_params.batch_size //= 4
                print('child_params.batch_size', child_params.batch_size)

            runner = run_sender.Runner()
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
                # 'val_same_acc': 'same_acc',
                # 'val_new_acc': 'new_acc'
            }
            res = {name_mapper.get(k, k): v for k, v in res.items()}
            res['send_architecture'] = send_architecture
            res['bsz'] = child_params.batch_size
            res['hg'] = hg
            print('res', res)
            res['t'] = str(int(res['t']))
            res['seed'] = child_seed
            res_keys = [
                'send_architecture', 'hg',
                'seed', 'b', 'bsz', 't', 'terminate_reason',
                'train_acc']
            # , 'same_acc', 'new_acc']
            for split_name in ['val_same', 'val_new', 'test_same', 'test_new']:
                for metric_name in ['acc']:
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
        '--sender-architectures', type=str,
        default='RNNOverCNN,RCNN,StackedInputs,MaxPoolingCNN,AveragePoolingCNN,PrototypicalSender')
    parser.add_argument(
        '--hgs', type=str,
        default='Colors1,Colors2,Colors3,Shapes1,Shapes2,Shapes3,Things1,Things2,Things3,Relations')

    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--out-csv-templ', type=str, default='{ref}.csv')
    parser.add_argument('--seed-base', type=int, default=123)
    # parser.add_argument('--tensor-dump-templ', type=str, default='tmp/{sub_ref}_{split_name}.pt')
    parser.add_argument('--max-mins', type=float, default=5, help='finish running if reach this elapsed time')

    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--save-every-seconds', type=int, default=-0)
    parser.add_argument('--render-every-seconds', type=int, default=30)
    parser.add_argument('--render-every-steps', type=int, default=-1)
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
    args.sender_architectures = args.sender_architectures.split(',')
    args.hgs = args.hgs.split(',')
    run(args)
