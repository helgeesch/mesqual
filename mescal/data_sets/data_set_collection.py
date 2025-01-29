from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Callable, Generic, Type, Iterable, TYPE_CHECKING

import pandas as pd

from mescal.data_sets.data_set import DataSet
from mescal.flag.flag_index import FlagIndex
from mescal.utils.pandas_utils.is_numeric import pd_is_numeric
from mescal.utils.logging import get_logger
from mescal.utils.pandas_utils.combine_df import combine_dfs
from mescal.utils.set_aggregations import nested_union
from mescal.utils.intersect_dicts import get_intersection_of_dicts
from mescal.typevars import DataSetType, DataSetConfigType, Flagtype

if TYPE_CHECKING:
    from mescal.kpis.kpi_base import KPIFactory
    from mescal.kpis.kpi_collection import KPICollection
    from mescal.databases.data_base import DataBase

logger = get_logger(__name__)


def _never_skip_ds_condition(ds: DataSet) -> bool:
    return False


def skip_ds_if_flag_not_accepted_condition(flag: Flagtype) -> Callable[[DataSet], bool]:
    return lambda ds: not ds.flag_is_accepted(flag)


class DataSetCollection(Generic[DataSetType, DataSetConfigType], DataSet[DataSetConfigType], ABC):
    """
    Abstract class to collect multiple DataSet instances
    and handle them according to a specific logic.
    Inherits all methods / functionalities from DataSet.
    """

    # Class attribute to store the data_set type,
    # can optionally be overwritten in child-classes
    data_set_type: Type[DataSetType] = DataSet

    def __init__(
            self,
            data_sets: list[DataSetType] = None,
            name: str = None,
            parent_data_set: DataSet = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            data_base: DataBase = None,
            config: DataSetConfigType = None
    ):
        super().__init__(
            name=name,
            parent_data_set=parent_data_set,
            flag_index=flag_index,
            attributes=attributes,
            data_base=data_base,
            config=config,
        )
        data_sets = data_sets if data_sets else []
        self.data_sets: dict[str, DataSetType] = {ds.name: ds for ds in data_sets}

    @property
    def flag_index(self) -> FlagIndex:
        from mescal.flag.flag_index import EmptyFlagIndex
        if (self._flag_index is None) or isinstance(self._flag_index, EmptyFlagIndex):
            from mescal.utils.check_all_same import all_same_object
            if all_same_object(ds.flag_index for ds in self.data_set_iterator) and len(self.data_sets):
                return self.get_data_set().flag_index
        return self._flag_index

    @property
    def attributes(self) -> pd.Series:
        atts = self._attributes.copy()
        child_data_set_atts = [ds.attributes.to_dict() for ds in self.data_sets.values()]
        attributes_that_all_childs_have_in_common = get_intersection_of_dicts(child_data_set_atts)
        for key in atts.keys():
            attributes_that_all_childs_have_in_common.pop(key, None)
        atts.update(attributes_that_all_childs_have_in_common)
        return pd.Series(atts, name=self.name)

    @property
    def attributes_df(self) -> pd.DataFrame:
        return pd.concat(
            {ds.name: ds.attributes for ds in self.data_set_iterator},
            axis=1,
            names=['data_set']
        ).rename_axis('attribute').T

    def get_kpi_df(self, **kwargs) -> pd.DataFrame:
        # TODO: concat axis
        # TODO: uniform order of magnitude per KPI / KPI category
        return pd.concat(
            {ds.name: ds.get_kpi_series(**kwargs) for ds in self.data_set_iterator},
            axis=1,
            names=['data_set']
        )

    def get_merged_kpi_collection(self) -> 'KPICollection':
        from mescal.kpis.kpi_collection import KPICollection
        all_kpis = set()
        for ds in self.data_set_iterator:
            for kpi in ds.kpi_collection:
                all_kpis.add(kpi)
            if isinstance(ds, DataSetCollection):
                for kpi in ds.get_merged_kpi_collection():
                    all_kpis.add(kpi)

        return KPICollection(all_kpis)

    def add_kpis_to_all_sub_data_sets(self, kpis: Iterable[KPIFactory]):
        for kpi in kpis:
            self.add_kpi_to_all_sub_data_sets(kpi)

    def add_kpi_to_all_sub_data_sets(self, kpi: KPIFactory):
        for ds in self.data_set_iterator:
            ds.add_kpi(kpi)

    def clear_kpi_collection_for_all_sub_data_sets(self):
        for ds in self.data_set_iterator:
            ds.clear_kpi_collection()

    @abstractmethod
    def _fetch(
            self,
            flag: Flagtype,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        pass

    def flag_is_accepted(self, flag: Flagtype) -> bool:
        return any(ds.flag_is_accepted(flag) for ds in self.data_set_iterator)

    @property
    def accepted_flags(self) -> set[Flagtype]:
        return nested_union([ds.accepted_flags for ds in self.data_set_iterator])

    def _required_flags_for_flag(self, flag: Flagtype) -> set[Flagtype]:
        return nested_union([ds.accepted_flags for ds in self.data_set_iterator])

    @property
    def data_set_iterator(self) -> Iterator[DataSetType]:
        for ds in self.data_sets.values():
            yield ds

    def get_data_set(self, key: str = None) -> DataSetType:
        if key is None:
            key = list(self.data_sets.keys())[0]
        return self.data_sets[key]

    def add_data_sets(self, data_sets: Iterable[DataSetType]):
        for ds in data_sets:
            self.add_data_set(ds)

    def add_data_set(self, data_set: DataSetType):
        if not isinstance(data_set, self.data_set_type):
            raise TypeError(f"Can only add data sets of type {self.data_set_type.__name__}.")
        if data_set.name not in self.data_sets:
            self.data_sets[data_set.name] = data_set
        else:
            logger.warning(
                f"DataSet {self.name}: "
                f"data_set {data_set.name} already in {type(self).__name__}. Not added again."
            )


class DataSetLinkCollection(Generic[DataSetType, DataSetConfigType], DataSetCollection[DataSetType, DataSetConfigType]):
    """
    Links multiple DataSet instances so:
        - the parent DataSet accepts flags of all child DataSets.
        - the child DataSet instances can fetch from each other.
    """

    def __init__(
            self,
            data_sets: list[DataSetType],
            name: str = None,
            parent_data_set: DataSet = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            data_base: DataBase = None,
            config: DataSetConfigType = None,
    ):
        super().__init__(
            data_sets=data_sets,
            name=name,
            parent_data_set=parent_data_set,
            flag_index=flag_index,
            attributes=attributes,
            data_base=data_base,
            config=config,
        )
        self._warn_if_flags_overlap()

    def _fetch(self, flag: Flagtype, **kwargs) -> pd.Series | pd.DataFrame:
        for ds in self.data_set_iterator:
            if ds.flag_is_accepted(flag):
                return ds.fetch(flag, **kwargs)
        raise KeyError(f"Key '{flag}' not recognized by any of the linked DataSets.")

    def _warn_if_flags_overlap(self):
        from collections import Counter

        accepted_flags = list()
        for ds in self.data_set_iterator:
            accepted_flags += list(ds.accepted_flags)

        counts = Counter(accepted_flags)
        duplicates = {k: v for k, v in counts.items() if v > 1}
        if any(duplicates.values()):
            logger.warning(
                f"DataSet {self.name}: "
                f"The following keys have multiple DataSet sources: {duplicates.keys()}. \n"
                f"Only the first one will be used! This might lead to unexpected behavior. \n"
                f"A potential reason could be the use of an inappropriate DataSetCollection Type."
            )


class DataSetMergeCollection(Generic[DataSetType, DataSetConfigType], DataSetCollection[DataSetType, DataSetConfigType]):
    """
    Fetch method will merge fragmented DataSets for same flag, e.g.:
        - fragmented simulation runs, e.g. CW1, CW2, CW3, CWn.
        - fragmented data sources, e.g. mapping from Excel file with model from simulation platform.
    """
    def __init__(
            self,
            data_sets: list[DataSetType],
            name: str = None,
            parent_data_set: DataSet = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            data_base: DataBase = None,
            config: DataSetConfigType = None,
            keep_first: bool = True,
    ):
        super().__init__(
            data_sets=data_sets,
            name=name,
            parent_data_set=parent_data_set,
            flag_index=flag_index,
            attributes=attributes,
            data_base=data_base,
            config=config,
        )
        self.keep_first = keep_first

    def _fetch(self, flag: Flagtype, **kwargs) -> pd.Series | pd.DataFrame:
        df = self._combine_dfs(
            get_df_from_data_set_method=lambda ds: ds.fetch(flag, **kwargs),
            keep_first=self.keep_first,
            skip_ds_condition=skip_ds_if_flag_not_accepted_condition(flag),
        )
        return df

    def _combine_dfs(
            self,
            get_df_from_data_set_method: Callable[[DataSetType], pd.Series | pd.DataFrame],
            skip_ds_condition: Callable[[DataSetType], bool] = None,
            keep_first: bool = True,
    ) -> pd.Series | pd.DataFrame:
        if skip_ds_condition is None:
            skip_ds_condition = _never_skip_ds_condition
        df = combine_dfs(
            [
                get_df_from_data_set_method(ds)
                for ds in self.data_set_iterator
                if not skip_ds_condition(ds)
            ],
            keep_first=keep_first
        )
        return df


class DataSetConcatCollection(Generic[DataSetType, DataSetConfigType], DataSetCollection[DataSetType, DataSetConfigType]):
    """
    Fetch method will return a concatenation of all sub-DataSets with an additional Index-level.
    """
    def __init__(
            self,
            data_sets: list[DataSetType],
            name: str = None,
            parent_data_set: DataSet = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            data_base: DataBase = None,
            config: DataSetConfigType = None,
            default_concat_axis: int = 1,
            concat_top: bool = True,
            concat_level_name: str = None,
    ):
        super().__init__(
            data_sets=data_sets,
            name=name,
            parent_data_set=parent_data_set,
            flag_index=flag_index,
            attributes=attributes,
            data_base=data_base,
            config=config,
        )
        super().__init__(data_sets=data_sets, name=name)
        self.default_concat_axis = default_concat_axis
        self.concat_top = concat_top
        self.concat_level_name = concat_level_name if concat_level_name is not None else 'data_set'

    def get_kpi_df(self, **kwargs) -> pd.DataFrame:
        # TODO: concat axis
        # TODO: uniform order of magnitude per KPI / KPI category
        return pd.concat(
            {ds.name: ds.get_kpi_series(**kwargs) for ds in self.data_set_iterator},
            axis=1,
            names=[self.concat_level_name]
        )

    def _fetch(
            self,
            flag: Flagtype,
            skip_ds_condition: Callable[[DataSet], bool] = None,
            concat_axis: int = None,
            transpose: bool = False,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        if concat_axis is None:
            concat_axis = self.default_concat_axis

        if skip_ds_condition is None:
            skip_ds_condition = skip_ds_if_flag_not_accepted_condition(flag)

        df = self._concat_data_set_dfs(
            get_df_from_data_set_method=lambda ds: ds.fetch(flag, **kwargs),
            skip_ds_condition=skip_ds_condition,
            axis=concat_axis,
            transpose=transpose
        )
        return df

    def _concat_data_set_dfs(
            self,
            get_df_from_data_set_method: Callable[[DataSetType], pd.Series | pd.DataFrame],
            skip_ds_condition: Callable[[DataSetType], bool] = None,
            axis: int = 1,
            transpose: bool = False,
    ) -> pd.Series | pd.DataFrame:
        if skip_ds_condition is None:
            skip_ds_condition = _never_skip_ds_condition
        dfs = {
            ds.name: get_df_from_data_set_method(ds)
            for ds in self.data_set_iterator
            if not skip_ds_condition(ds)
        }

        if len(dfs):
            df0 = list(dfs.values())[0]
            if not all(len(df.axes) == len(df0.axes) for df in dfs.values()):
                raise NotImplementedError(f'Axes lengths do not match between dfs.')
            for ax in range(len(list(dfs.values())[0].axes)):
                if not all(set(df.axes[ax].names) == set(df0.axes[ax].names) for df in dfs.values()):
                    raise NotImplementedError(f'Axes names do not match between dfs.')

            # from mescal.utils.pandas_utils.standardize_indices import standardize_index
            # dfs = standardize_index(dfs, axis=axis)

        df = pd.concat(dfs, join='outer', axis=axis, names=[self.concat_level_name])

        if not self.concat_top:
            ax: pd.MultiIndex = df.axes[axis]
            df.axes[axis] = ax.reorder_levels([ax.nlevels] + list(range(ax.nlevels - 1)))

        if transpose:
            return df.transpose()
        return df


class DataSetSumCollection(Generic[DataSetType, DataSetConfigType], DataSetCollection[DataSetType, DataSetConfigType]):
    def _fetch(self, flag: Flagtype, **kwargs) -> pd.Series | pd.DataFrame:
        data: list[pd.Series | pd.DataFrame] = []
        for ds in self.data_set_iterator:
            if ds.flag_is_accepted(flag):
                data.append(ds.fetch(flag, **kwargs))
        if not data:
            raise KeyError(f"Flag '{flag}' not recognized by any of the linked DataSets.")
        
        if all(pd_is_numeric(d) for d in data):
            import numpy as np
            return np.sum(data)

        raise NotImplementedError
