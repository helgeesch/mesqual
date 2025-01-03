from mescal.data_sets.data_set import DataSet
from mescal.flag.flag import Flagtype


def get_all_intersecting_objects_from_related_model_flag(
        data_set: DataSet,
        variable_flag: Flagtype,
        model_query: str,
) -> list[str]:

    model_flag = data_set.flag_index.get_linked_model_flag(variable_flag)
    model_df = data_set.fetch(model_flag)
    if model_query is not None:
        model_df = model_df.query(model_query, engine='python')

    variable_df = data_set.fetch(variable_flag)
    if variable_df.columns.nlevels > 1:
        raise NotImplementedError  # TODO: identify correct column level
    objects_in_variable_df = list(variable_df.columns.get_level_values(0).unique())

    return list(set(model_df.index).intersection(objects_in_variable_df))
