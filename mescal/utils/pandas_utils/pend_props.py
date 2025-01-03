from typing import TypeGuard
import pandas as pd


def is_dataframe(obj: pd.Series | pd.DataFrame) -> TypeGuard[pd.DataFrame]:
    return isinstance(obj, pd.DataFrame)


def is_series(obj: pd.Series | pd.DataFrame) -> TypeGuard[pd.Series]:
    return isinstance(obj, pd.Series)


def get_matching_axis_and_level(data: pd.Series | pd.DataFrame, match_index_level: pd.Index) -> tuple[int, int]:
    if isinstance(match_index_level, pd.MultiIndex):
        raise ValueError('Method only works for single index level.')

    missing_values = dict()
    for axis in range(len(data.axes)):
        for level in range(data.axes[axis].nlevels):
            missing = set(data.axes[axis].get_level_values(level)).difference(match_index_level)
            if len(missing) == 0:
                return axis, level
            missing_values[(axis, level)] = len(missing)

    raise ValueError(f"No Index Match Found. Missing values: {missing_values}")


def prepend_model_prop_levels(
        data: pd.Series | pd.DataFrame,
        model: pd.DataFrame,
        *properties,
        prepend_to_top: bool = True,
        match_on_level: str = None,
) -> pd.Series | pd.DataFrame:
    """
    Searches for an index level in df to match with model_df.
    It will then prepend the properties as new index levels to df.
    """
    tmp = data.copy()
    properties = [p for p in properties if not ((p is None) or (p == ''))]

    if not properties:
        return tmp

    if match_on_level is not None:
        if isinstance(match_on_level, int):
            raise NotImplementedError
        if match_on_level in tmp.columns.names:
            axis = 1
        elif match_on_level in tmp.index.names:
            axis = 0
        else:
            raise ValueError
        level = list(tmp.axes[axis].names).index(match_on_level)
    else:
        for prop in properties:
            if prop not in model.columns.tolist():
                raise ValueError(f'Property unavailable: {prop} was not found in your model_df.')
        axis, level = get_matching_axis_and_level(data, model.index)

    match_keys = tmp.axes[axis].get_level_values(level)
    new_index = tmp.axes[axis].to_frame(index=False)
    for prop in properties:
        if prop not in new_index:
            loc = 0 if prepend_to_top else len(new_index.columns)
            new_index.insert(loc, prop, model.loc[match_keys, prop].values)
    new_index = pd.MultiIndex.from_frame(new_index)
    if axis == 0:
        tmp.index = new_index
    else:
        tmp.columns = new_index

    if is_series(data):
        tmp: pd.Series = tmp
        return tmp
    elif is_dataframe(data):
        tmp: pd.DataFrame = tmp
        return tmp
    else:
        return tmp
