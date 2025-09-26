import pandas as pd


def identify_and_convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies columns that contain only True / False / empty values and converts them to boolean dtype.
    """
    for column in df:
        unique_values = df[column].dropna().unique()
        if len(set(unique_values).difference({True, False})) == 0:
            df.loc[df[column].isna(), column] = 0
            df = df.astype({column: bool})
    return df
