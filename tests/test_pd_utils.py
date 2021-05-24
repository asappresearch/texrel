import pytest
import pandas as pd

from ulfs import pd_utils


@pytest.mark.parametrize(
    "df,row_name,column_name,metric_name,expected",
    [
        (
            pd.DataFrame({
                'send_arch': ['prot', 'prot', 'stack', 'stack'],
                'tasks': ['colors', 'rels', 'colors', 'rels'],
                'val_acc': [3, 4, 5.2, 6.4]
            }),
            'send_arch', 'tasks', 'val_acc',
            pd.DataFrame({
                'send_arch': ['prot', 'stack'],
                'colors': [3, 5.2],
                'rels': [4, 6.4],
            })
        ),
    ]
)
def test_do_pivot(df, row_name, column_name, metric_name, expected):
    pivoted = pd_utils.do_pivot(
        df=df, row_name=row_name, col_name=column_name, metric_name=metric_name
    )
    print('df\n' + str(df))
    print('pivoted\n' + str(pivoted))
    print('pivoted\n' + str(pivoted))
    print('expected\n' + str(expected))
    pd.testing.assert_frame_equal(pivoted, expected)
