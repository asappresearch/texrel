"""
reduce runs comparing shapeworld and texrel
we will have scenarios as rows, and we'll be maximizing over columns

no pivoting required
"""
from typing import List
import argparse
import pandas as pd

from ulfs import stats_utils, tex_utils


def reduce(df_l: List[pd.DataFrame], out_csv: str, out_tex: str):
    # drop_columns = [
    #     'seed', 'b', 'terminate_reason', 'train_acc',
    #     'val_same_gnd_clusters', 'val_same_pred_clusters', 'val_new_gnd_clusters', 'val_new_pred_clusters']
    keep_columns = [
        'ds_family', 'sampler_model', 't', 'augment',
        'train_acc',
        'test_same_acc', 'test_same_rho', 'test_same_prec', 'test_same_rec',
        'test_new_acc', 'test_new_rho', 'test_new_prec', 'test_new_rec']
    # df_l = [df.drop(columns=drop_columns) for df in df_l]
    df_l = [df[keep_columns].copy() for df in df_l]
    for i, df in enumerate(list(df_l)):
        df = df.rename(columns={'sampler_model': 'train_sampler'})
        columns = list(df.columns)
        print('columns', columns)
        columns = ['code'] + columns
        df['code'] = df.ds_family
        df.t = df.t / 60
        df.loc[df.code != 'lsl_shapeworld', 'code'] = 'ours'
        df.loc[df.code == 'lsl_shapeworld', 'code'] = 'LSL'
        df.ds_family = df.ds_family.str.replace('lsl_', '').replace('texrel', '\\textsc{TexRel}').replace(
            'shapeworld', 'Shapeworld')
        df['test_sampler'] = df.train_sampler.copy()
        df.loc[df.code == 'LSL', 'train_sampler'] = 'soft'
        df.train_sampler = df.train_sampler.str.replace('Softmax', 'soft').replace('Gumbel', 'gumb')
        df.test_sampler = df.test_sampler.str.replace('Softmax', 'soft').replace('Gumbel', 'discr').replace(
            'discrete', 'discr')
        columns = columns[:3] + ['test_sampler'] + columns[3:]
        # df.loc[(df.code == 'lsl_shapeworld') & (df.augment), 'ds_family'] = 'Shapeworld+aug'
        df.loc[(df.code == 'LSL') & (df.augment), 'ds_family'] = 'Shapeworld+aug'
        df = df[columns]
        df = df.drop(columns=['augment'])
        df_l[i] = df
    averaged_str_no_ci, average, ci, counts = stats_utils.average_of_dfs(df_l, show_ci=False, show_mean=True, max_sig=2)
    averaged_str_with_ci, average, ci, counts = stats_utils.average_of_dfs(
        df_l, show_ci=False, show_mean=True, max_sig=2)
    averaged_str_with_ci.t = averaged_str_no_ci.t
    print(averaged_str_with_ci)

    averaged_str_with_ci.to_csv(out_csv)

    highlight = tex_utils.get_best_by_column(
        df_str_no_ci=averaged_str_no_ci, df_mean=average, maximize=[
            'train_acc',
            'test_same_acc', 'test_same_rho', 'test_same_prec', 'test_same_rec',
            'test_new_acc', 'test_new_rho', 'test_new_prec', 'test_new_rec'
        ]
    )
    titles = {
        'ds family': 'Dataset',
        'code': 'Code',
        't': 'Time',
        'train sampler': 'Train sampler',
        'test sampler': 'Test sampler',
        'train acc': '$\\acc_{train}$',

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
        caption='Comparison between \\textsc{{TexRel}} and ShapeWorld datasets for emergent communications scenario',
        label='tab:vs_shapeworld')


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
