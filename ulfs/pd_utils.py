import pandas as pd


def do_pivot(df: pd.DataFrame, row_name: str, col_name: str, metric_name: str):
    """
    Works with df.pivot, except preserves the ordering of the rows and columns
    in the pivoted dataframe
    """
    original_row_indices = df[row_name].unique()
    original_col_indices = df[col_name].unique()
    pivoted = df.pivot(index=row_name, columns=col_name, values=metric_name)
    pivoted = pivoted[original_col_indices]
    pivoted = pivoted.reindex(original_row_indices).reset_index()
    pivoted.columns.name = None
    return pivoted
