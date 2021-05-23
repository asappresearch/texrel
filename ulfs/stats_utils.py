from typing import Tuple, Sequence

import numpy as np
import pandas as pd

from ulfs import formatting


def average_of_dfs(
        dfs: Sequence[pd.DataFrame], show_ci: bool, show_mean: bool, max_sig: int = 3) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Takes in list of dataframes
    for any numeric columns, takes average. other columns
    verifies identical

    returns results as strings, averages, 95% CIs, and counts
    """
    sums = dfs[0].copy()
    counts = dfs[0].copy()
    fields = dfs[0].columns
    types = dfs[0].dtypes
    key_fields = list(dfs[0].index.names)
    print('key_fields', key_fields)
    numeric_fields = [field for field, dtype in zip(fields, types) if dtype in [
        np.float32, np.int64, np.float64, np.int32] and field not in key_fields]
    print('numeric_fields', numeric_fields)
    sums[numeric_fields] = 0.0
    counts[numeric_fields] = 0

    np_values_l = []
    for df in dfs:
        sums[numeric_fields] = sums[numeric_fields] + df[numeric_fields].fillna(0)
        counts[numeric_fields] = counts[numeric_fields] + df[numeric_fields].notnull().astype('int')
        values = df[numeric_fields].values
        np_values_l.append(values)
    np_values = np.stack(np_values_l)
    np_stddev = np.nanstd(np_values, axis=0)
    stddev = dfs[0].copy()
    stddev[numeric_fields] = np_stddev
    ci = dfs[0].copy()
    ci[numeric_fields] = stddev[numeric_fields] / counts[numeric_fields].pow(0.5) * 1.96

    average = sums.copy()
    average[numeric_fields] = sums[numeric_fields] / counts[numeric_fields]
    average[numeric_fields] = average[numeric_fields].astype(np.float32)

    # average = average.set_index(key_fields)
    # ci = ci.set_index(key_fields)

    averaged_str = dfs[0].copy()
    averaged_str[numeric_fields] = averaged_str[numeric_fields].astype(str)
    for index, average_row in average.iterrows():
        ci_row = ci.loc[index]
        averaged_str_row = averaged_str.loc[index]
        for field in numeric_fields:
            averaged_str_row[field] = formatting.mean_err_to_str(
                average_row[field], ci_row[field].item(), err_sds=1, max_sig=max_sig,
                show_err=show_ci, show_mean=show_mean, na_val='')
    # average = average.reset_index()
    return averaged_str, average, ci, counts
