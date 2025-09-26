from typing import TypeGuard
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
        data: pd.Series | pd.DataFrame,
        match_index_level: pd.Index,
        match_on_level: int | str = None
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

    raise ValueError(f"No Index Match Found. Missing values: {missing_values}")


def prepend_model_prop_levels(
        data: pd.Series | pd.DataFrame,
        model: pd.DataFrame,
        *properties,
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

    Energy Domain Context:
        In Energy Systems Analysis, you often have to groupby and aggregate
        by certain properties. This module makes it easy to include the properties
        as a new index level before performing the groupby - agg pipeline.

    Example:

        >>> # You have a generation time-series df
        >>> print(gen_df)  # Original DataFrame
            generator            GenA  GenB  GenC  SolarA  WindA
            2024-01-01 00:00:00   100   200   150      50     80
            2024-01-01 01:00:00   120   180   170      60     90
            2024-01-01 02:00:00   110   190   160      55     85

        >>> # You have a generator model df
        >>> print(model_df)
                      zone technology  is_res
            generator
            GenA        DE    nuclear   False
            GenB        DE       coal   False
            GenC        FR        gas   False
            SolarA      DE      solar    True
            WindA       NL       wind    True

        >>> gen_with_props = prepend_model_prop_levels(gen_df, model_df, 'zone', 'is_res')
        >>> print(gen_with_props)  # DataFrame with prepended properties
            is_res              False            True
            zone                   DE        FR     DE    NL
            generator            GenA GenB GenC SolarA WindA
            2024-01-01 00:00:00   100  200  150     50    80
            2024-01-01 01:00:00   120  180  170     60    90
            2024-01-01 02:00:00   110  190  160     55    85

        >>> gen_by_zone_and_type = gen_with_props.T.groupby(level=['zone', 'is_res']).sum().T
        >>> print(gen_by_zone_and_type)  # grouped and aggregated
            zone                   DE          FR    NL
            is_res              False True  False True
            2024-01-01 00:00:00   300    50   150    80
            2024-01-01 01:00:00   300    60   170    90
            2024-01-01 02:00:00   300    55   160    85
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


if __name__ == '__main__':
    import numpy as np

    # Example 1: Basic usage with DataFrame and Series
    print("Example 1: Basic usage")
    print("=" * 50)

    # Create model with generator properties
    model_data = {
        'generator': ['GenA', 'GenB', 'GenC', 'SolarA', 'WindA'],
        'zone': ['DE', 'DE', 'FR', 'DE', 'NL'],
        'technology': ['nuclear', 'coal', 'gas', 'solar', 'wind'],
        'is_res': [False, False, False, True, True]
    }
    model_df = pd.DataFrame(model_data).set_index('generator')
    print("Model DataFrame:")
    print(model_df)
    print()

    # Create simple Series with generator data
    prices_series = pd.Series([50, 45, 55, 0, 0],
                             index=pd.Index(['GenA', 'GenB', 'GenC', 'SolarA', 'WindA'],
                                           name='generator'))
    print("Original Series:")
    print(prices_series)
    print()

    # Prepend properties to Series
    prices_with_props = prepend_model_prop_levels(prices_series, model_df, 'zone', 'technology')
    print("Series with prepended properties:")
    print(prices_with_props)
    print()

    # Example 2: DataFrame with time series data
    print("Example 2: DataFrame with time dimension")
    print("=" * 50)

    # Create model with generator properties
    model_df = pd.DataFrame(
        {
            'generator': ['GenA', 'GenB', 'GenC', 'SolarA', 'WindA'],
            'zone': ['DE', 'DE', 'FR', 'DE', 'NL'],
            'technology': ['nuclear', 'coal', 'gas', 'solar', 'wind'],
            'is_res': [False, False, False, True, True]
        }
    ).set_index('generator')

    # Create DataFrame with time and generators
    gen_df = pd.DataFrame(
        np.array([
            [100, 200, 150, 50, 80],
            [120, 180, 170, 60, 90],
            [110, 190, 160, 55, 85]
        ]),
        index=pd.date_range('2024-01-01', periods=3, freq='h'),
        columns=pd.Index(['GenA', 'GenB', 'GenC', 'SolarA', 'WindA'], name='generator')
    )
    print(gen_df)  # Original DataFrame

    gen_with_props = prepend_model_prop_levels(gen_df, model_df, 'zone', 'is_res')
    print(gen_with_props)  # DataFrame with prepended properties

    gen_by_zone_and_type = gen_with_props.T.groupby(level=['zone', 'is_res']).sum().T
    print(gen_by_zone_and_type)  # grouped and aggregated
