import pandas as pd


def merge_multi_index_levels(
        multi_index: pd.MultiIndex,
        levels: list[str],
        name_of_new_level: str,
        join_levels_by: str = ' - ',
        append_new_level_as_last: bool = True
) -> pd.MultiIndex:
    """Merge multiple levels of a MultiIndex into a single new level.

    Combines specified levels from a pandas MultiIndex by joining their values
    with a separator string, creating a new level while preserving other levels.

    Args:
        multi_index: The MultiIndex to modify.
        levels: List of level names to merge together.
        name_of_new_level: Name for the newly created merged level.
        join_levels_by: String used to join the level values. Defaults to ' - '.
        append_new_level_as_last: If True, append new level at the end. If False,
            prepend at the beginning. Defaults to True.

    Returns:
        A new MultiIndex with the specified levels merged into a single level.

    Examples:

        >>> import pandas as pd
        >>> index = pd.MultiIndex.from_tuples([
        ...     ('DE', 'solar', '2024'),
        ...     ('DE', 'wind', '2024'),
        ...     ('FR', 'nuclear', '2024')
        ... ], names=['country', 'technology', 'year'])
        >>>
        >>> # Merge country and technology levels
        >>> new_index = merge_multi_index_levels(
        ...     index,
        ...     ['country', 'technology'],
        ...     'location_tech',
        ...     join_levels_by='_'
        ... )
        >>> print(new_index.names)
            ['year', 'location_tech']
        >>> print(new_index.tolist())
            [('2024', 'DE_solar'), ('2024', 'DE_wind'), ('2024', 'FR_nuclear')]
    """
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