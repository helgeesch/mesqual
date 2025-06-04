import pandas as pd


def merge_multi_index_levels(
        multi_index: pd.MultiIndex,
        levels: list[str],
        name_of_new_level: str,
        join_levels_by: str = ' - ',
        append_new_level_as_last: bool = True
) -> pd.MultiIndex:
    df_index = multi_index.to_frame()
    merged_level = df_index[levels].astype(str).agg(join_levels_by.join, axis=1)
    remaining_levels = [level for level in multi_index.names if level not in levels]

    if append_new_level_as_last:
        new_level_order = remaining_levels + [name_of_new_level]
    else:
        new_level_order = [name_of_new_level] + remaining_levels

    df_index = df_index[remaining_levels]
    df_index[name_of_new_level] = merged_level

    return pd.MultiIndex.from_frame(df_index[new_level_order])


if __name__ == '__main__':
    index = pd.MultiIndex.from_tuples([
        ('DE', 'solar', '2024'),
        ('DE', 'wind', '2024'),
        ('FR', 'nuclear', '2024')
    ], names=['country', 'technology', 'year'])

    df = pd.DataFrame(index=index)
    new_index = merge_multi_index_levels(df.index, ['technology', 'country'], 'asset', join_levels_by=' - ')
    print(new_index)