"""
Run various pairs of architectures, and see which ones learn best,
and give highest compositional measures

test something like:
python ref_task/runners/run_arch_e2e_comparison.py --disable-cuda --ref foo --sampler-model Gumbel \
    --hgs Relations --max-steps 10 --ds-collection macdebug --batch-size 8
python ref_task/runners/run_arch_e2e_comparison.py --ref foo --max-mins 0.05 --batch-size 32 \
    --sampler-model Gumbel --hgs Relations
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


def run(args):
    results = []
    for send_arch in args.senders:
        for recv_arch in args.receivers:
            for hg in args.hgs:
                print('')
                print('=======================================')
                print('send=', send_arch, 'recv=', recv_arch, 'hg=', hg)
                child_params = Params(dict(args.__dict__))
                child_params.ref = f'{args.ref}_{send_arch}_{recv_arch}_{hg}'
                child_params.image_seq_embedder = send_arch
                child_params.multimodal_classifier = recv_arch
                child_params.ds_tasks = hg
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
                res['send_arch'] = send_arch
                res['recv_arch'] = recv_arch
                res['bsz'] = child_params.batch_size
                res['hg'] = hg
                print('res', res)
                res['t'] = str(int(res['t']))
                res['seed'] = child_seed
                res_keys = [
                    'send_arch', 'recv_arch', 'hg',
                    'seed', 'b', 'bsz', 't',
                    'terminate_reason',
                    'train_acc']
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

    # parser.add_argument(
    #     '--arch-pairs', type=str,
    #     default='PrototypicalSender:PrototypicalReceiver,'
    #             'RCNN:LearnedCNNMapping')
    parser.add_argument(
        '--senders', type=str,
        default='StackedInputs,PrototypicalSender')
    parser.add_argument(
        '--receivers', type=str,
        default='AllPlaneAttention,Cosine,FeaturePlaneAttention')

    parser.add_argument(
        '--hgs', type=str,
        # default='Colors1,Colors2,Colors3,Shapes1,Shapes2,Shapes3,Things1,Things2,Things3,Relations')
        default='Relations')

    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--out-csv-templ', type=str, default='{ref}.csv')
    parser.add_argument('--seed-base', type=int, default=123)
    # parser.add_argument('--tensor-dump-templ', type=str, default='tmp/{sub_ref}_{split_name}.pt')
    parser.add_argument('--max-mins', type=float, default=5, help='finish running if reach this elapsed time')

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
    # args.arch_pairs = args.arch_pairs.split(',')
    args.senders = args.senders.split(',')
    args.receivers = args.receivers.split(',')
    args.hgs = args.hgs.split(',')
    run(args)
