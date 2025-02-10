import pandas as pd

from mescal.utils.pandas_utils.pend_props import get_matching_axis_and_level


def filter_by_model_query(
        df: pd.Series | pd.DataFrame,
        model_df: pd.DataFrame,
        query: str = None,
        match_on_level: int | str = None,
) -> pd.DataFrame | pd.Series:
    if query is None or query == '':
        return df

    axis, idx_selection_level = get_matching_axis_and_level(df, model_df.index, match_on_level)
    idx = df.axes[axis]

    selection = model_df.query(query, engine='python').copy(deep=True).index
    selection = list(set(selection).intersection(idx.get_level_values(idx_selection_level)))

    if isinstance(idx, pd.MultiIndex):
        selection = [i for i in idx if i[idx_selection_level] in selection]

    if isinstance(df, pd.Series):
        return df[selection] if axis == 0 else df.loc[selection]
    else:
        return df.loc[selection, :] if axis == 0 else df.loc[:, selection]


if __name__ == '__main__':
    import numpy as np
    model_data = {
        'zone': ['DE', 'FR', 'NL', 'BE'],
        'price': [50, 45, 55, 60],
        'has_renewable': [True, False, True, False]
    }
    model_df = pd.DataFrame(model_data).set_index('zone')

    timeindex = pd.date_range('2024-01-01', '2024-01-02', freq='h')
    zones = ['DE', 'FR', 'NL', 'BE']
    flows = np.random.randint(0, 1000, size=(len(timeindex), len(zones) * len(zones)))

    cols = pd.MultiIndex.from_product([zones, zones], names=['from_zone', 'to_zone'])
    flows_df = pd.DataFrame(flows, index=timeindex, columns=cols)

    prices_series = pd.Series([50, 45, 55, 60], index=pd.Index(['DE', 'FR', 'NL', 'BE'], name='zone'))
    volumes_series = pd.Series([1000, 2000, 1500, 3000], index=pd.MultiIndex.from_tuples(
        [('DE', 'peak'), ('FR', 'base'), ('NL', 'peak'), ('BE', 'base')],
        names=['zone', 'load_type']
    ))

    filtered_df = filter_by_model_query(flows_df, model_df, 'has_renewable == True', match_on_level='from_zone')
    filtered_simple_series = filter_by_model_query(prices_series, model_df, 'price > 50')
    filtered_multi_series = filter_by_model_query(volumes_series, model_df, 'has_renewable == False')
