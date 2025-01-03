import pandas as pd


def _get_ordered_set(my_list: list) -> list:
    d = dict.fromkeys(my_list)
    return list(d.keys())


def standardize_index(dfs: dict[str, pd.DataFrame], axis: int) -> dict[str, pd.DataFrame]:

    all_names = [lvl_name for df in dfs.values() for lvl_name in df.axes[axis].names]
    all_names = _get_ordered_set(all_names)

    if len(all_names) <= 1:
        return dfs

    # TODO: handle dataframes without level_names

    for df_name, df in dfs.items():
        existing_names = set(df.axes[axis].names)
        missing_names = set(all_names) - existing_names

        for lvl_name in missing_names:
            idx = df.axes[axis]
            idx_new = pd.MultiIndex.from_product(
                [
                    i + ('',) if len(idx.names) > 1 else (i, '')
                    for i in idx
                ],
                names=list(idx.names) + [lvl_name]
            )
            df.axes[axis] = idx_new
        df.axes[axis] = df.axes[axis].reorder_levels(all_names)

    return dfs
