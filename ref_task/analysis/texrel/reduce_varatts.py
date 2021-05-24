"""
given output of multiple runs of run_numatts_numvalues.py, calculates their
means, and cis and outputs these

but we also need to pivot etc the data into a form that we can plot the various
graphs we need in pgfplots
"""
import argparse
import csv
import pandas as pd

from ulfs import stats_utils


def run(args):
    df_l = []
    type_by_numeric_col = {
        'b': 'int', 't': 'int', 'num_entities': 'int', 'num_values': 'int',
        'rho': 'float32', 'prec': 'float32', 'rec': 'float32',
        'gnd_cls': 'int', 'pred_cls': 'int',
        'train_acc': 'float32', 'same_acc': 'float32', 'new_acc': 'float32'
    }
    for in_csv in args.in_csvs:
        with open(in_csv, 'r') as f:
            dict_reader = csv.DictReader(f)
            rows = list(dict_reader)
        df = pd.DataFrame(rows)
        df = df.astype(type_by_numeric_col)
        df = df.set_index(['entity', 'num_entities', 'num_values'])
        df_l.append(df)
    min_length = min([len(df) for df in df_l])
    print('min_length', min_length)
    max_length = max([len(df) for df in df_l])
    print('max_length', max_length)
    if min_length != max_length:
        print('WARNING: not all files same length')
        df_l = [df[:min_length] for df in df_l]
    for df in df_l:
        print(df)

    formatted_strs, mean, ci95, counts = stats_utils.average_of_dfs(
        df_l, show_ci=True, show_mean=True)

    print('formatted strings:')
    print(formatted_strs)
    print('')

    print('mean')
    print(mean)
    print('')

    print('95% CI')
    print(ci95)
    print('')

    print('counts')
    print(counts)

    df_output = mean.copy()
    df_output.reset_index()
    ci95 = ci95.reset_index()
    for col_name in type_by_numeric_col.keys():
        df_output[f'{col_name}_ci95'] = ci95[col_name]
    print('output')
    print(df_output)
    print(df_output.iloc[0])

    df_by_num_values = pd.DataFrame({'num_values': list(range(3, 10))})
    df_by_num_values = df_by_num_values.set_index('num_values')
    df_mean_no_index = mean.reset_index()
    ci95_no_index = ci95.reset_index()
    for i, entity in enumerate(['colors', 'shapes', 'things']):
        df_entity = df_mean_no_index[df_mean_no_index.entity == entity]
        for num_entities in [1, 2, 3]:
            df_ent_num = df_entity[df_entity.num_entities == num_entities]
            df_ent_num = df_ent_num.set_index(['num_values'])
            for value in ['rho', 'prec', 'rec', 'same_acc', 'new_acc']:
                df_by_num_values[f'{entity}{num_entities}_{value}'] = df_ent_num[value]
    for i, entity in enumerate(['colors', 'shapes', 'things']):
        df_entity = ci95_no_index[ci95_no_index.entity == entity]
        for num_entities in [1, 2, 3]:
            df_ent_num = df_entity[df_entity.num_entities == num_entities]
            df_ent_num = df_ent_num.set_index(['num_values'])
            for value in ['rho', 'prec', 'rec', 'same_acc', 'new_acc']:
                df_by_num_values[f'{entity}{num_entities}_{value}_ci95'] = df_ent_num[f'{value}']
    print('df_by_num_values', df_by_num_values)
    df_by_num_values.to_csv(args.out_csv_wide)

    df_output.to_csv(args.out_csv_long)

    print('')
    if min_length != max_length:
        print('WARNING: not all files same length')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csvs', type=str, required=True, nargs='+', help='paths to csv input files')
    parser.add_argument('--out-csv-long', type=str, required=True)
    parser.add_argument('--out-csv-wide', type=str, required=True)
    args = parser.parse_args()
    run(args)
