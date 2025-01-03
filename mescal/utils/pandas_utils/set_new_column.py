from typing import Hashable
import pandas as pd


def set_column(
        df: pd.DataFrame,
        new_column_name: Hashable,
        new_column_values: pd.Series | pd.DataFrame
) -> pd.DataFrame:

    if not len(df) == len(new_column_values):
        raise ValueError('Length of df and new_column_values must be equal.')

    # TODO optional: check index

    if isinstance(new_column_values, pd.Series):
        df[new_column_name] = new_column_values
        return df

    if isinstance(new_column_values, pd.DataFrame):
        if not new_column_values.columns.nlevels == (df.columns.nlevels - 1):
            raise ValueError(
                'Your new_column_values must have n-1 column levels, where n is the number of levels in df.'
            )

        if new_column_name in df.columns:
            df = df.drop(columns=[new_column_name])

        new_column_values = pd.concat({new_column_name: new_column_values}, axis=1, names=[df.columns.names[0]])
        df = pd.concat([df, new_column_values], axis=1)
        return df

    else:
        raise TypeError('Used new_column_values type not accepted.')