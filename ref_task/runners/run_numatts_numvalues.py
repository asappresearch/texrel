"""
grid over number of attributes, and number of attribute values
ie 2 shapes, with 3 different kinds of shapes, etc

for dev run, something like:

python ref_task/runners/run_numatts_numvalues.py --ref foo --render-every-seconds 1 --max-steps 100 \
    --batch-size 32 --sampler-model Gumbel
"""
import argparse
import csv
from os import path
from os.path import expanduser as expand

from ulfs.params import Params
from ref_task import params_groups, run_end_to_end


ds_ref_templ_num_attributes = {
    3: 'ords056sc3-{entity}{num_entities}2',
    4: 'ords055sc4-{entity}{num_entities}2',
    5: 'ords057sc5-{entity}{num_entities}2',
    6: 'ords058sc6-{entity}{num_entities}2',
    7: 'ords059sc7-{entity}{num_entities}2',
    8: 'ords060sc8-{entity}{num_entities}2',
    9: 'ords061sc9-{entity}{num_entities}2',
}


class FakeRunner(object):
    def __init__(self, parser):
        self.parser = parser

    def add_param(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)


def run(args):
    results = []
    for num_attributes in range(3, 10):
        for num_entities in [1, 2, 3]:
            for entity in ['colors', 'shapes', 'things']:
                if entity in ['colors', 'shapes'] and num_attributes <= 3 and num_entities == 3:
                    continue
                child_params = Params(dict(args.__dict__))
                child_params.ref = f'{args.ref}_a{num_attributes}_e{num_entities}_{entity}'
                # del child_params.__dict__['fast_dev_run']
                child_seed = args.seed_base + 0
                child_params.seed = child_seed
                del child_params.__dict__['seed_base']

                ds_ref = ds_ref_templ_num_attributes[num_attributes].format(
                    entity=entity, num_entities=num_entities
                )
                print('ds_ref', ds_ref)
                child_params.ds_collection = None
                child_params.ds_tasks = None
                child_params.ds_distractors = None
                child_params.ds_refs = ','.join([ds_ref])

                child_params.tensor_dumps_templ_path_on_terminate = args.tensor_dump_templ.format(
                    sub_ref=child_params.ref, split_name='{split_name}')
                del child_params.__dict__['tensor_dump_templ']

                runner = run_end_to_end.Runner()
                runner._extract_standard_args(child_params)
                runner.enable_cuda = child_params.enable_cuda
                if not path.exists(expand(
                        child_params.ds_filepath_templ.format(ds_ref=ds_ref))):
                    print('')
                    print('**********************************************************')
                    print('**** data file for', ds_ref, ' not exists => skipping ****')
                    print('**********************************************************')
                    print('')
                    continue
                runner.setup_base(params=child_params)
                runner.run_base()
                res = runner.res
                print('res', res)

                name_mapper = {
                    'batch': 'b',
                    'elapsed_time': 't',
                    'acc': 'train_acc',
                    # 'test_same_rho': 'rho',
                    # 'test_same_prec': 'prec',
                    # 'test_same_rec': 'rec',
                    # 'test_same_gnd_clusters': 'gnd_cls',
                    # 'test_same_pred_clusters': 'pred_cls',
                    # 'test_same_acc': 'same_acc',
                    # 'test_new_acc': 'new_acc'
                }
                res = {name_mapper.get(k, k): v for k, v in res.items()}
                res['num_values'] = num_attributes
                res['num_entities'] = num_entities
                res['entity'] = entity
                print('res', res)
                res['t'] = str(int(res['t']))
                res['seed'] = child_seed
                res_keys = [
                    'entity', 'num_entities', 'num_values', 'seed', 'b', 't', 'train_acc']
                # 'rho', 'prec', 'rec', 'gnd_cls', 'pred_cls',
                # 'train_acc', 'same_acc', 'new_acc']
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

    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--out-csv-templ', type=str, default='{ref}.csv')
    parser.add_argument('--seed-base', type=int, default=123)
    parser.add_argument('--tensor-dump-templ', type=str, default='tmp/{sub_ref}_{split_name}.pt')

    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--save-every-seconds', type=int, default=-0)
    parser.add_argument('--render-every-seconds', type=int, default=30)
    parser.add_argument('--render-every-steps', type=int, default=-1)
    parser.add_argument('--name', type=str, default='run_numatts_numvalues', help='used for logfile naming')
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
