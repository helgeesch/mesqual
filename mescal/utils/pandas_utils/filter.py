import pandas as pd

from mescal.utils.pandas_utils.pend_props import get_matching_axis_and_level


def filter_by_model_query(
        df: pd.Series | pd.DataFrame,
        model: pd.DataFrame,
        query: str = None,
        query_engine: str = 'python'
) -> pd.DataFrame:
    """
    This function allows you to filter the dataframe data based on a query in your model.The index of your model
    must exist either as a level in your data.axes[axis].
    """
    if isinstance(df, pd.Series):
        raise NotImplementedError  # TODO: should also work for pd.Series

    if query is None or query == '':
        return df

    axis, idx_selection_level = get_matching_axis_and_level(df, model.index)
    idx = df.axes[axis]

    selection = model.query(query, engine=query_engine).copy(deep=True).index
    selection = list(set(selection).intersection(idx.get_level_values(idx_selection_level)))

    if isinstance(idx, pd.MultiIndex):
        selection = [i for i in idx if i[idx_selection_level] in selection]
    if axis == 0:
        return df.loc[selection, :]
    else:
        return df.loc[:, selection]
