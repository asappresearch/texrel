"""
reduce multitask results
involves a pivot, since we want the scenarios along the top, and the hg down the side
"""
from typing import List
import argparse
import pandas as pd

from ulfs import stats_utils, tex_utils, pd_utils


def reduce(df_l: List[pd.DataFrame], metric_name: str, out_csv: str, out_tex: str):
    # scenarios = df_l[0].scenario.unique()
    # print('scenarios', scenarios)

    # hgs = df_l[0].hg.unique()
    # print('hgs', hgs)

    df_l = [pd_utils.do_pivot(
        df=df, row_name='hg', col_name='scenario', metric_name=metric_name) for df in df_l]
    averaged_str_no_ci, average, ci, counts = stats_utils.average_of_dfs(df_l, show_ci=False, show_mean=True, max_sig=2)
    averaged_str_with_ci, average, ci, counts = stats_utils.average_of_dfs(
        df_l, show_ci=True, show_mean=True, max_sig=2)
    # averaged_str_with_ci[averaged_str_with_ci.hg == 'Things3']['hg'] = 'SCs3'
    averaged_str_with_ci.loc[averaged_str_with_ci.hg == 'Things3', 'hg'] = 'SCs3'
    print(averaged_str_with_ci)

    averaged_str_with_ci.to_csv(out_csv)

    metric_name_display = metric_name.replace('_', ' ')

    highlight = tex_utils.get_best_by_row(
        df_str_no_ci=averaged_str_no_ci, df_mean=average
    )

    titles = {
        'hg': 'Task',
        'none': 'No multitask',
        'colors12': 'Colors1,Colors2',
        'shapes12': 'Shapes1,Shapes2',
        'things12': 'SCs1,SCs2',
        'all': 'all',
        'allalldists': 'all, +0 dists'
    }

    tex_utils.write_tex(
        df_str=averaged_str_with_ci, df_mean=average, filepath=out_tex,
        highlight=highlight,
        titles=titles,
        add_arrows=False,
        caption=f'Multitask {metric_name_display}',
        label=f'tab:multitask_{metric_name}')


def run(args):
    df_l = []
    for ref in args.in_refs:
        df = pd.read_csv(f'pull/{ref}.csv')
        df_l.append(df)
    for metric_name in ['test_same_acc', 'test_new_acc', 'test_same_rho', 'test_new_rho']:
        reduce(
            df_l,
            metric_name,
            out_csv=args.out_csv_templ.format(out_ref=args.out_ref, metric_name=metric_name),
            out_tex=args.out_tex_templ.format(out_ref=args.out_ref, metric_name=metric_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-refs', type=str, nargs='+')
    parser.add_argument('--out-ref', type=str, required=True)
    parser.add_argument('--out-csv-templ', type=str, default='pull/{out_ref}_{metric_name}.csv')
    parser.add_argument('--out-tex-templ', type=str, default='pull/{out_ref}_{metric_name}.tex')
    args = parser.parse_args()
    run(args)
