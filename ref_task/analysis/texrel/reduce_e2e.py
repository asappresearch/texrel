"""
so, we basically dont have to do much. just take the average over the runs, find the max, and write
to latex format
"""
from typing import List
import argparse
import pandas as pd

from ulfs import stats_utils, tex_utils


def reduce(df_l: List[pd.DataFrame], out_csv: str, out_tex: str):
    keep_columns = [
        'send_arch', 'recv_arch', 'b', 't',
        'train_acc',
        'test_same_acc', 'test_same_rho', 'test_same_prec', 'test_same_rec',
        'test_new_acc', 'test_new_rho', 'test_new_prec', 'test_new_rec']
    df_l = [df[keep_columns].copy() for df in df_l]
    sender_name_mapping = {
        'PrototypicalSender': 'Prototypical'
    }
    receiver_name_mapping = {
        'AllPlaneAttention': 'AllPlaneAtt',
        'FeaturePlaneAttention': 'FeatPlaneAtt'
    }
    for df in df_l:
        df.t = df.t / 60
        df.b = df.b / 1000
        # df.b = df.b.map('{:}k'.format)
        for src, dest in sender_name_mapping.items():
            df.send_arch = df.send_arch.str.replace(src, dest)
        for src, dest in receiver_name_mapping.items():
            df.recv_arch = df.recv_arch.str.replace(src, dest)
    averaged_str_no_ci, average, ci, counts = stats_utils.average_of_dfs(df_l, show_ci=False, show_mean=True, max_sig=2)
    averaged_str_with_ci, average, ci, counts = stats_utils.average_of_dfs(
        df_l, show_ci=False, show_mean=True, max_sig=2)
    averaged_str_with_ci.t = averaged_str_no_ci.t
    print(averaged_str_with_ci)

    averaged_str_with_ci.b = averaged_str_with_ci.b.map('{:}k'.format)

    averaged_str_with_ci.to_csv(out_csv)

    highlight = tex_utils.get_best_by_column(
        df_str_no_ci=averaged_str_no_ci, df_mean=average, maximize=[
            'train_acc',
            'test_same_acc', 'test_same_rho', 'test_same_prec', 'test_same_rec',
            'test_new_acc', 'test_new_rho', 'test_new_prec', 'test_new_rec'
        ]
    )
    titles = {
        'send arch': 'Sender',
        'recv arch': 'Receiver',
        't': 'Time',
        # 'train sampler': 'Train sampler',
        # 'sampler model': 'Test sampler',
        # 'train acc': '$\\acc_{train}$',

        'test same acc': '$\\acc_{same}$',
        'test same rho': '$\\rho_{same}$',
        'test same prec': '$\\text{prec}_{same}$',
        'test same rec': '$\\text{rec}_{same}$',

        'test new acc': '$\\acc_{new}$',
        'test new rho': '$\\rho_{new}$',
        'test new prec': '$\\text{prec}_{new}$',
        'test new rec': '$\\text{rec}_{new}$',
    }
    tex_utils.write_tex(
        df_str=averaged_str_with_ci, df_mean=average, filepath=out_tex,
        highlight=highlight,
        titles=titles,
        latex_defines='\\def\\acc{\\text{acc}}',
        add_arrows=False,
        caption=(
            'Comparison on end-to-end architectures. Each result is mean over 5 runs. '
            'Utterances are sampled from Gumbel distributions. '
            'The underlying task is Relations. Early stopping on $\\acc_{val\\_same}$'),
        label='tab:e2e_comp')


def run(args):
    df_l = []
    for ref in args.in_refs:
        df = pd.read_csv(f'pull/{ref}.csv')
        df_l.append(df)
    reduce(
        df_l,
        out_csv=args.out_csv_templ.format(out_ref=args.out_ref),
        out_tex=args.out_tex_templ.format(out_ref=args.out_ref))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-refs', type=str, nargs='+')
    parser.add_argument('--out-ref', type=str, required=True)
    parser.add_argument('--out-csv-templ', type=str, default='pull/{out_ref}.csv')
    parser.add_argument('--out-tex-templ', type=str, default='pull/{out_ref}.tex')
    args = parser.parse_args()
    run(args)
