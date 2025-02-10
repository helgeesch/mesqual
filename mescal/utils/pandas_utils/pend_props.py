from typing import TypeGuard
import warnings
import pandas as pd

def is_dataframe(obj: pd.Series | pd.DataFrame) -> TypeGuard[pd.DataFrame]:
    return isinstance(obj, pd.DataFrame)


def is_series(obj: pd.Series | pd.DataFrame) -> TypeGuard[pd.Series]:
    return isinstance(obj, pd.Series)


def get_matching_axis_and_level(
        data: pd.Series | pd.DataFrame,
        match_index_level: pd.Index,
        match_on_level: int | str = None
) -> tuple[int, int]:
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
    """
    Searches for an index level in df to match with model_df.
    It will then prepend the properties as new index levels to df.
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
    model_data = {
        'generator': ['GenA', 'GenB', 'GenC', 'GenD', 'GenE', 'GenF', 'SolarA', 'SolarB', 'WindA'],
        'zone': ['DE', 'DE', 'FR', 'NL', 'BE', 'BE', 'DE', 'FR', 'NL'],
        'marginal_cost': [50, 48, 45, 55, 60, 62, 0, 0, 0],
        'is_res': [False, False, False, False, False, False, True, True, True],
        'technology': ['nuclear', 'nuclear', 'coal', 'gas', 'nuclear', 'gas', 'solar', 'solar', 'wind']
    }
    model_df = pd.DataFrame(model_data).set_index('generator')

    timeindex = pd.date_range('2024-01-01', '2024-01-02', freq='h')
    gens = model_df.index.tolist()
    flows = np.random.randint(0, 1000, size=(len(timeindex), len(gens)))

    flows_df = pd.DataFrame(flows, index=timeindex, columns=pd.Index(gens, name='generator'))
    prices_series = pd.Series(model_df.marginal_cost.values, index=pd.Index(gens, name='generator'))

    flows_with_props = prepend_model_prop_levels(flows_df, model_df, 'zone', 'technology', 'is_res')
    prices_with_props = prepend_model_prop_levels(prices_series, model_df, 'zone', 'is_res')
