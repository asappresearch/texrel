"""
reduce over send results

rows are architectures, and we are maximizing over columns
"""
from typing import List
import argparse
import pandas as pd

from ulfs import stats_utils, tex_utils, pd_utils


def reduce(df_l: List[pd.DataFrame], metric_name: str, out_csv: str, out_tex: str):

    hgs = df_l[0].hg.unique()
    print('hgs', hgs)

    row_name = f'{args.direction}_architecture'
    df_l = [pd_utils.do_pivot(
        df=df, row_name=row_name, col_name='hg', metric_name=metric_name) for df in df_l]
    averaged_str, average, ci, counts = stats_utils.average_of_dfs(df_l, show_ci=False, show_mean=True, max_sig=2)
    print(averaged_str)

    averaged_str.to_csv(out_csv)

    titles = {
        'Colors1': 'col1',
        'Colors2': 'col2',
        'Colors3': 'col3',
        'Shapes1': 'shp1',
        'Shapes2': 'shp2',
        'Shapes3': 'shp3',
        'Things1': 'sc1',
        'Things2': 'sc2',
        'Things3': 'sc3',
        'Relations': 'rels',
    }
    if args.direction == 'recv':
        titles['recv architecture'] = 'Receiver architecture'
    else:
        titles['send architecture'] = 'Sender architecture'
    highlight = tex_utils.get_best_by_column(
        df_str_no_ci=averaged_str,
        df_mean=average,
        maximize=hgs
    )
    tex_utils.write_tex(
        df_str=averaged_str,
        df_mean=average,
        filepath=out_tex,
        highlight=highlight,
        titles=titles,
        add_arrows=False,
        caption=f'{args.direction} {metric_name}',
        label=f'tab:{args.direction}_{metric_name}')


def run(args):
    df_l = []
    for ref in args.in_refs:
        df = pd.read_csv(f'pull/{ref}.csv')
        df_l.append(df)
    reduce(df_l, 'same_acc', out_csv=args.out_same_acc, out_tex=args.out_same_acc_tex)
    reduce(df_l, 'new_acc', out_csv=args.out_new_acc, out_tex=args.out_new_acc_tex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-refs', type=str, nargs='+')
    parser.add_argument('--direction', type=str, choices=['send', 'recv'])
    parser.add_argument('--out-ref', type=str, required=True)
    parser.add_argument('--out-same-acc', type=str, default='pull/{out_ref}_same_acc.csv')
    parser.add_argument('--out-new-acc', type=str, default='pull/{out_ref}_new_acc.csv')
    parser.add_argument('--out-same-acc-tex', type=str, default='pull/{out_ref}_same_acc.tex')
    parser.add_argument('--out-new-acc-tex', type=str, default='pull/{out_ref}_new_acc.tex')
    args = parser.parse_args()
    args.out_same_acc = args.out_same_acc.format(out_ref=args.out_ref)
    args.out_new_acc = args.out_new_acc.format(out_ref=args.out_ref)
    args.out_same_acc_tex = args.out_same_acc_tex.format(out_ref=args.out_ref)
    args.out_new_acc_tex = args.out_new_acc_tex.format(out_ref=args.out_ref)
    run(args)
