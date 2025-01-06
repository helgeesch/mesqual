from __future__ import annotations

from typing import TYPE_CHECKING, Union, Type, Iterable, Generic
from abc import ABC, abstractmethod

import pandas as pd

from mescal.databases.data_base import DataBase
from mescal.utils.string_conventions import to_lower_snake
from mescal.flag.flag_index import EmptyFlagIndex, FlagIndex
from mescal.utils.logging import get_logger
from mescal.typevars import DataSetConfigType, Flagtype

if TYPE_CHECKING:
    from mescal.data_sets.data_set_collection import DataSetLinkCollection
    from mescal.kpis.kpi_collection import KPICollection
    from mescal.kpis.kpi_base import KPI, KPIFactory

logger = get_logger(__name__)


def flag_must_be_accepted(method):
    def raise_if_flag_not_accepted(self: DataSet, flag: Flagtype = None, **kwargs):
        if not self.flag_is_accepted(flag):
            raise ValueError(f'Flag {flag} not accepted by DataSet "{self.name}" of type {type(self)}.')
        return method(self, flag, **kwargs)
    return raise_if_flag_not_accepted


def return_from_db_if_possible(method):
    def _return_data_from_db_if_possible(self: DataSet, flag: Flagtype = None, **kwargs):
        if self._data_base is None:
            return method(self, flag, **kwargs)

        key = self._get_key_for_db(flag, **kwargs)
        if self._data_base.key_is_up_to_date(key, **kwargs):
            return self._data_base.get(key, **kwargs)

        data = method(self, flag, **kwargs)
        self._data_base.set(key, data, **kwargs)
        return data

    return _return_data_from_db_if_possible


def ensure_unique_indices(method):
    def _remove_duplicate_indices_if_there_are_any(self: DataSet, flag: Flagtype = None, **kwargs):
        data = method(self, flag, **kwargs)
        if any(data.index.duplicated()):
            logger.info(
                f'For some reason your data-set {self.name} returns an object with duplicate indices for flag {flag}.\n'
                f'We manually remove duplicate indices. Please make sure your data importer / converter is set up '
                f'appropriately and that your raw data does not contain duplicate indices. \n'
                f'We will keep the first element of every duplicated index.'
            )
            data = data.loc[~data.index.duplicated()]
        return data
    return _remove_duplicate_indices_if_there_are_any


class DataSet(Generic[DataSetConfigType], ABC):
    def __init__(
            self,
            name: str = None,
            parent_data_set: DataSet = None,
            flag_index: FlagIndex = None,
            attributes: dict = None,
            data_base: DataBase = None,
            config: DataSetConfigType = None
    ):
        self.name = name if name is not None else self._get_random_name()
        self._flag_index = flag_index or EmptyFlagIndex()
        self._parent_data_set = parent_data_set
        self._attributes: dict = attributes if attributes is not None else dict()
        self._data_base = data_base
        self._config = config

        from mescal.kpis.kpi_collection import KPICollection
        self.kpi_collection: KPICollection = KPICollection()

    def _get_key_for_db(self, flag: Flagtype, **kwargs) -> str:
        return f'{self.name} {flag}'

    @property
    def flag_index(self) -> FlagIndex:
        if isinstance(self._flag_index, EmptyFlagIndex):
            logger.info(
                f"DataSet {self.name}: "
                "You're trying to use functionality of the FlagIndex but didn't define one. "
                "The current FlagIndex in use is empty. "
                "Make sure to set a flag_index in case you want to use full functionality of the flag_index."
            )
        return self._flag_index

    def add_kpis(self, kpis: Iterable[KPI | KPIFactory | Type[KPI]]):
        for kpi in kpis:
            self.add_kpi(kpi)

    def add_kpi(self, kpi: KPI | KPIFactory | Type[KPI]):
        from mescal.kpis.kpis_from_aggregations import KPIFactory
        if isinstance(kpi, KPIFactory):
            kpi = kpi.get_kpi(self)
        elif isinstance(kpi, type) and issubclass(kpi, KPI):
            kpi = kpi.from_factory(self)
        self.kpi_collection.add_kpi(kpi)

    def get_kpi_series(self, **kwargs) -> pd.Series:
        return self.kpi_collection.get_kpi_series(**kwargs)

    @property
    def attributes(self) -> pd.Series:
        att_series = pd.Series(self._attributes, name=self.name)
        return att_series

    def set_attributes(self, **kwargs):
        self._attributes.update(kwargs)

    @property
    def parent_data_set(self) -> 'DataSetLinkCollection':
        if self._parent_data_set is None:
            raise RuntimeError(f"Parent data_set called without / before assignment.")
        return self._parent_data_set

    @parent_data_set.setter
    def parent_data_set(self, parent_data_set: 'DataSetLinkCollection'):
        from mescal.data_sets.data_set_collection import DataSetLinkCollection
        if not isinstance(parent_data_set, DataSetLinkCollection):
            raise TypeError(f"Parent data_set must be of type DataSet")
        self._parent_data_set = parent_data_set

    @property
    @abstractmethod
    def accepted_flags(self) -> set[Flagtype]:
        return set()

    def flag_is_accepted(self, flag: Flagtype) -> bool:
        """
        This method can be optionally overridden in any child-class
        in case you want to follow logic instead of the explicit set of accepted_flags.
        """
        return flag in self.accepted_flags

    @flag_must_be_accepted
    def required_flags_for_flag(self, flag: Flagtype) -> set[Flagtype]:
        return self._required_flags_for_flag(flag)

    @abstractmethod
    def _required_flags_for_flag(self, flag: Flagtype) -> set[Flagtype]:
        return set()

    @flag_must_be_accepted
    @return_from_db_if_possible
    @ensure_unique_indices
    def fetch(self, flag: Flagtype, **kwargs) -> pd.Series | pd.DataFrame:
        return self._fetch(flag, **kwargs).copy()

    @abstractmethod
    def _fetch(self, flag: Flagtype, **kwargs) -> pd.Series | pd.DataFrame:
        return pd.DataFrame()

    def fetch_multiple_flags_and_concat(
            self,
            flags: Iterable[Flagtype],
            concat_axis: int = 1,
            concat_level_name: str = 'variable',
            concat_level_at_top: bool = True,
            **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        dfs = {
            flag: self.fetch(flag, **kwargs)
            for flag in flags
        }
        df = pd.concat(
            dfs,
            axis=concat_axis,
            names=[concat_level_name],
        )
        if not concat_level_at_top:
            ax = df.axes[concat_axis]
            ax = ax.reorder_levels(list(range(1, ax.nlevels)) + [0])
            df.axes[concat_axis] = ax
        return df

    @classmethod
    def _get_class_name_lower_snake(cls) -> str:
        return to_lower_snake(cls.__name__)

    @classmethod
    def _get_random_name(cls) -> str:
        import random
        import string

        _num_chars = 6
        part_1 = cls._get_class_name_lower_snake()
        part_2 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=_num_chars))
        return f'{part_1}_{part_2}'

    def fetch_filter_groupby_agg(
            self,
            flag: Flagtype,
            model_filter_query: str = None,
            prop_groupby: str | list[str] = None,
            prop_groupby_agg: str = None,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        model_flag = self.flag_index.get_linked_model_flag(flag)
        if not model_flag:
            raise RuntimeError(f'FlagIndex could not successfully map flag {flag} to a model flag.')

        from mescal.utils import pandas_utils

        data = self.fetch(flag, **kwargs)
        model_df = self.fetch(model_flag, **kwargs)

        if model_filter_query:
            data = pandas_utils.filter_by_model_query(data, model_df, query=model_filter_query)

        if prop_groupby:
            if isinstance(prop_groupby, str):
                prop_groupby = [prop_groupby]
            data = pandas_utils.prepend_model_prop_levels(data, model_df, *prop_groupby)
            data = data.groupby(prop_groupby)
            if prop_groupby_agg:
                data = data.agg(prop_groupby_agg)
        elif prop_groupby_agg:
            logger.warning(
                f"You provided a prop_groupby_agg operation, but didn't provide prop_groupby. "
                f"No aggregation performed."
            )
        return data

    @classmethod
    def get_config_type(cls) -> Type[DataSetConfigType]:
        raise NotImplementedError("Subclasses must implement get_config_type")

    @property
    def config(self) -> DataSetConfigType:
        from mescal.data_sets.data_set_config import DataSetConfigManager
        return DataSetConfigManager.get_effective_config(self.__class__, self._config)

    def set_config(self, config: DataSetConfigType) -> None:
        self._config = config

    def set_config_kwargs(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self._config, key, value)

    @classmethod
    def set_class_config(cls, config: DataSetConfigType) -> None:
        from mescal.data_sets.data_set_config import DataSetConfigManager
        DataSetConfigManager.set_class_config(cls, config)
