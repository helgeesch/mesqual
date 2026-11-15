"""Prepend model properties as index levels to data."""

from typing import Hashable, TypeGuard
import warnings
import pandas as pd


def is_dataframe(obj: pd.Series | pd.DataFrame) -> TypeGuard[pd.DataFrame]:
    """Check if object is a pandas DataFrame.

    Args:
        obj: Object to check, expected to be either Series or DataFrame.

    Returns:
        True if obj is a DataFrame, False otherwise.
    """
    return isinstance(obj, pd.DataFrame)


def is_series(obj: pd.Series | pd.DataFrame) -> TypeGuard[pd.Series]:
    """Check if object is a pandas Series.

    Args:
        obj: Object to check, expected to be either Series or DataFrame.

    Returns:
        True if obj is a Series, False otherwise.
    """
    return isinstance(obj, pd.Series)


def get_matching_axis_and_level(
    data: pd.Series | pd.DataFrame, match_index_level: pd.Index, match_on_level: int | str = None
) -> tuple[int, int]:
    """Find the axis and level in data that matches the given index level.

    Searches through all axes and levels of the data to find where the index values
    match the provided match_index_level. Returns the first match found.

    Args:
        data: The pandas object to search within.
        match_index_level: Index to match against data's axes.
        match_on_level: Optional constraint to match on specific level (by position or name).

    Returns:
        Tuple of (axis, level) indicating where the match was found.

    Raises:
        ValueError: If match_index_level is a MultiIndex, or if no match is found.

    Warnings:
        UserWarning: If multiple matches found and no explicit match_on_level provided.
    """
    if isinstance(match_index_level, pd.MultiIndex):
        raise ValueError('Method only works for single index level.')

    missing_values = dict()
    matches = []
    for axis in range(len(data.axes)):
        for level in range(data.axes[axis].nlevels):
            missing = set(data.axes[axis].get_level_values(level)).difference(match_index_level)
            if len(missing) == 0:
                if match_on_level is None:
                    matches.append((axis, level))
                elif isinstance(match_on_level, int) and (level == match_on_level):
                    matches.append((axis, level))
                elif isinstance(match_on_level, str) and (data.axes[axis].names[level] == match_on_level):
                    matches.append((axis, level))
            missing_values[(axis, level)] = len(missing)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        warnings.warn(
            f'Multiple levels match your index, but no explicit match_on_level was provided. '
            f'Auto fall-back first match; axis, level: {matches[0]}'
        )
        return matches[0]

    raise ValueError(f'No Index Match Found. Missing values: {missing_values}')


def prepend_model_prop_levels(
    data: pd.Series | pd.DataFrame,
    model: pd.DataFrame,
    *properties: Hashable,
    prepend_to_top: bool = True,
    match_on_level: str = None,
) -> pd.Series | pd.DataFrame:
    """Prepend model properties as new index levels to data.

    Searches for an index level in data that matches the model's index, then
    prepends specified properties from the model as new index levels.

    Args:
        data: The pandas object to add properties to.
        model: DataFrame containing properties to prepend, with matching index.
        *properties: Column names from model to use as new index levels.
        prepend_to_top: If True, add properties at the beginning of index levels.
            If False, add at the end.
        match_on_level: Optional level name to constrain matching to specific level.
            Useful in case the there are multiple index levels in data that match
            the model's index

    Returns:
        Copy of data with properties prepended as new index levels.

    Raises:
        ValueError: If any property is not found in model columns.

    Use it before a groupby to aggregate by a model property, for example
    the generation per node or per fuel.

    Examples:
        >>> generation = dataset.fetch('Generator.Results.generation')
        >>> generators = dataset.fetch('Generator.Model')
        >>> with_props = prepend_model_prop_levels(generation, generators, 'node', 'fuel')
        >>> with_props.columns.tolist()
        [('nuclear', 'N1', 'G1'), ('coal', 'N1', 'G2'), ('gas', 'N2', 'G3'), ('wind', 'N2', 'G4'), ('solar', 'N3', 'G5')]
        >>> with_props.T.groupby(level='node').sum().T.iloc[:2].round(1)
        node                           N1     N2   N3
        time
        2024-03-28 00:00:00+01:00  1218.4  891.8  0.0
        2024-03-28 01:00:00+01:00  1247.7  825.7  0.0
    """
    tmp = data.copy()
    properties = [p for p in properties if not ((p is None) or (p == ''))]

    if not properties:
        return tmp

    for prop in properties:
        if prop not in model.columns.tolist():
            raise ValueError(f'Property unavailable: {prop} was not found in your model_df.')
    axis, level = get_matching_axis_and_level(data, model.index, match_on_level)

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
