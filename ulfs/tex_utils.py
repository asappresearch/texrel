import os
from typing import Dict, Iterable, Set, Tuple
import pandas as pd
import numpy as np


def get_best_by_column(
        df_str_no_ci: pd.DataFrame, df_mean: pd.DataFrame, maximize: Iterable[str] = (), minimize: Iterable[str] = ()):
    highlight = set()
    N = len(df_str_no_ci)
    print('N', N)
    maximize = [field for field in maximize if field in df_str_no_ci.columns]
    minimize = [field for field in minimize if field in df_str_no_ci.columns]
    for field in list(maximize) + list(minimize):
        if field in maximize:
            best_index = df_mean[field].argmax()
        else:
            best_index = df_mean[field].argmin()
        best_value = df_str_no_ci[field][best_index]
        matches_value = df_str_no_ci[field] == best_value
        for i, matches in enumerate(matches_value):
            if matches:
                highlight.add((i, field))
    return highlight


def get_best_by_row(
        df_str_no_ci: pd.DataFrame, df_mean: pd.DataFrame):
    """
    for now this always maximizes...
    """
    N = len(df_str_no_ci)
    print('N', N)

    highlight = set()

    fields = df_mean.columns
    types = df_mean.dtypes
    key_fields = list(df_mean.index.names)
    numeric_fields = [field for field, dtype in zip(fields, types) if dtype in [
        np.float32, np.int64, np.float64, np.int32] and field not in key_fields
        and field != 'index']

    best_field_by_row = df_mean[numeric_fields].idxmax(axis=1)
    for n, field in enumerate(best_field_by_row):
        highlight.add((n, field))
    return highlight


def write_tex(
        df_str: pd.DataFrame, df_mean: pd.DataFrame, filepath: str,
        highlight: Set[Tuple[int, str]],
        longtable: bool = False,
        titles: Dict[str, str] = {},
        latex_defines: str = '',
        caption: str = '',
        label: str = '',
        add_arrows: bool = True) -> None:
    """
    - maximize and minimize are columns we want to find the max/min of and make bold
    - maximize rows are rows we want to find the max of and make bold
    """
    df_str = df_str.reset_index()
    df_mean = df_mean.reset_index()

    # caption = caption.replace('_', '\\_')

    fieldnames = list(df_str.columns)
    if 'index' in fieldnames:
        fieldnames.remove('index')

    with open(filepath, 'w') as f:
        f.write("""
\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=0.5in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
""".format() + latex_defines)
        if longtable:
            f.write("""
\\usepackage{{longtable}}""".format())
        f.write("""
\\begin{{document}}
""".format())

        if longtable:
            f.write("""
{{
\\small
\\begin{{longtable}}[l]{{ {alignment} }}
""".format(alignment='l' * len(fieldnames)))
        else:
            f.write("""
\\begin{{table*}}[htb!]
\\small
\\centering
\\begin{{tabular}}{{ {alignment} }}""".format(alignment='l' * len(fieldnames)))

        f.write("""
\\toprule
""".format())

        row_l = []
        for field in fieldnames:
            field = field.replace('_', ' ')
            field_str = titles.get(field, field)
            row_l.append(field_str)
        f.write(' & '.join(row_l) + ' \\\\ \n')
        f.write('\\midrule \n')
        for n, row in enumerate(df_str.to_dict('records')):
            row_l = []
            for field in fieldnames:
                value_str = str(row[field]).replace('_', ' ')
                if (n, field) in highlight:
                    value_str = f'\\textbf{{{value_str}}}'
                row_l.append(value_str)
            f.write(' & '.join(row_l) + ' \\\\ \n')
        f.write("""\\bottomrule""".format())

        if not longtable:
            f.write("""
\\end{{tabular}}""".format())

        f.write("""
\\caption{{{caption}}}
\\label{{{label}}}""".format(caption=caption, label=label))

        if longtable:
            f.write("""
\\end{{longtable}}
}}""".format())
        else:
            f.write("""
\\end{{table*}}
""".format())

        f.write("""
\\end{{document}}
""".format())

    os.system(f'open {filepath}')
