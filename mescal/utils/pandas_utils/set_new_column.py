from typing import Hashable
import pandas as pd


def set_column(
        df: pd.DataFrame,
        new_column_name: Hashable,
        new_column_values: pd.Series | pd.DataFrame
) -> pd.DataFrame:

    dff = df.copy()

    if not len(dff) == len(new_column_values):
        raise ValueError('Length of dff and new_column_values must be equal.')

    # TODO optional: check index

    if isinstance(new_column_values, pd.Series):
        dff[new_column_name] = new_column_values
        return dff

    if isinstance(new_column_values, pd.DataFrame):
        if not new_column_values.columns.nlevels == (dff.columns.nlevels - 1):
            raise ValueError(
                'Your new_column_values must have n-1 column levels, where n is the number of levels in dff.'
            )

        if new_column_name in dff.columns:
            dff = dff.drop(columns=[new_column_name])

        new_column_values = pd.concat({new_column_name: new_column_values}, axis=1, names=[dff.columns.names[0]])
        dff = pd.concat([dff, new_column_values], axis=1)
        return dff

    else:
        raise TypeError('Used new_column_values type not accepted.')