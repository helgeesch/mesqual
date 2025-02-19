from __future__ import annotations

from typing import TYPE_CHECKING, Union, Type, Iterable, Generic
from abc import ABC, abstractmethod

import pandas as pd

from mescal.typevars import DataSetConfigType, Flagtype
from mescal.databases.data_base import DataBase
from mescal.utils.string_conventions import to_lower_snake
from mescal.flag.flag_index import EmptyFlagIndex, FlagIndex
from mescal.utils.logging import get_logger

if TYPE_CHECKING:
    from mescal.data_sets.data_set_collection import DataSetLinkCollection
    from mescal.kpis.kpi_collection import KPICollection
    from mescal.kpis.kpi_base import KPI, KPIFactory

logger = get_logger(__name__)


def flag_must_be_accepted(method):
    def raise_if_flag_not_accepted(self: DataSet, flag: Flagtype = None, config: DataSetConfigType = None, **kwargs):
        if not self.flag_is_accepted(flag):
            raise ValueError(f'Flag {flag} not accepted by DataSet "{self.name}" of type {type(self)}.')
        return method(self, flag, config, **kwargs)
    return raise_if_flag_not_accepted


class _DotNotationFetcher:
    """
    Enables dot notation access for DataSet flag fetching.

    Accumulates flag parts through attribute access and converts them to a flag via
    the dataset's flag_index when executed. Supports both immediate execution through
    direct dataset attribute access and delayed execution through fetch_dotted.

    Usage:
        data_set.dotfetch.my.flag.as.string()
    """
    def __init__(self, data_set, accumulated_parts: list[str] = None):
        self._data_set = data_set
        self._accumulated_parts = accumulated_parts or []

    def __getattr__(self, part: str) -> '_DotNotationFetcher':
        return _DotNotationFetcher(self._data_set, self._accumulated_parts + [part])

    def __str__(self) -> str:
        return '.'.join(self._accumulated_parts)

    def __call__(self) -> pd.DataFrame | pd.Series:
        return self._data_set.fetch(self._data_set.flag_index.get_flag_from_string(str(self)))


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
        self.name = name or f'{self.__class__.__name__}_{str(id(self))}'
        self._flag_index = flag_index or EmptyFlagIndex()
        self._parent_data_set = parent_data_set
        self._attributes: dict = attributes if attributes is not None else dict()
        self._data_base = data_base
        self._config = config
        self.dotfetch = _DotNotationFetcher(self)

        from mescal.kpis.kpi_collection import KPICollection
        self.kpi_collection: KPICollection = KPICollection()

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
        from mescal.kpis.kpi_base import KPI
        from mescal.kpis.kpis_from_aggregations import KPIFactory
        if isinstance(kpi, KPIFactory):
            kpi = kpi.get_kpi(self)
        elif isinstance(kpi, type) and issubclass(kpi, KPI):
            kpi = kpi.from_factory(self)
        self.kpi_collection.add_kpi(kpi)

    def get_kpi_series(self, **kwargs) -> pd.Series:
        return self.kpi_collection.get_kpi_series(**kwargs)

    def clear_kpi_collection(self):
        from mescal.kpis import KPICollection
        self.kpi_collection = KPICollection()

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
            raise TypeError(f"Parent parent_data_set must be of type {DataSetLinkCollection.__name__}")
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
    def fetch(self, flag: Flagtype, config: dict | DataSetConfigType = None, **kwargs) -> pd.Series | pd.DataFrame:
        effective_config = self._prepare_config(config)
        use_database = self._data_base is not None and effective_config.use_database

        if use_database:
            if self._data_base.key_is_up_to_date(self, flag, config=effective_config, **kwargs):
                return self._data_base.get(self, flag, config=effective_config, **kwargs)

        raw_data = self._fetch(flag, config=effective_config, **kwargs)
        processed_data = self._post_process_data(raw_data, flag, effective_config)

        if use_database:
            self._data_base.set(self, flag, processed_data, config=effective_config, **kwargs)

        return processed_data.copy()

    def _post_process_data(
            self,
            data: pd.Series | pd.DataFrame,
            flag: Flagtype,
            config: DataSetConfigType
    ) -> pd.Series | pd.DataFrame:
        if config.remove_duplicate_indices and any(data.index.duplicated()):
            logger.info(
                f'For some reason your data-set {self.name} returns an object with duplicate indices for flag {flag}.\n'
                f'We manually remove duplicate indices. Please make sure your data importer / converter is set up '
                f'appropriately and that your raw data does not contain duplicate indices. \n'
                f'We will keep the first element of every duplicated index.'
            )
            data = data.loc[~data.index.duplicated()]
        if config.auto_sort_datetime_index and isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        return data

    def _prepare_config(self, config: dict | DataSetConfigType = None) -> DataSetConfigType:
        if config is None:
            return self.instance_config

        if isinstance(config, dict):
            temp_config = self.get_config_type()()
            temp_config.__dict__.update(config)
            return self.instance_config.merge(temp_config)

        from mescal.data_sets.data_set_config import DataSetConfig
        if isinstance(config, DataSetConfig):
            return self.instance_config.merge(config)

        raise TypeError(f"Config must be dict or {DataSetConfig.__name__}, got {type(config)}")

    @abstractmethod
    def _fetch(self, flag: Flagtype, config: dict | DataSetConfigType = None, **kwargs) -> pd.Series | pd.DataFrame:
        return pd.DataFrame()

    def fetch_multiple_flags_and_concat(
            self,
            flags: Iterable[Flagtype],
            concat_axis: int = 1,
            concat_level_name: str = 'variable',
            concat_level_at_top: bool = True,
            config: dict | DataSetConfigType = None,
            **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        dfs = {
            flag: self.fetch(flag, config, **kwargs)
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

    def fetch_filter_groupby_agg(
            self,
            flag: Flagtype,
            model_filter_query: str = None,
            prop_groupby: str | list[str] = None,
            prop_groupby_agg: str = None,
            config: dict | DataSetConfigType = None,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        model_flag = self.flag_index.get_linked_model_flag(flag)
        if not model_flag:
            raise RuntimeError(f'FlagIndex could not successfully map flag {flag} to a model flag.')

        from mescal.utils import pandas_utils

        data = self.fetch(flag, config, **kwargs)
        model_df = self.fetch(model_flag, config, **kwargs)

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
        from mescal.data_sets.data_set_config import DataSetConfig
        return DataSetConfig

    @property
    def instance_config(self) -> DataSetConfigType:
        from mescal.data_sets.data_set_config import DataSetConfigManager
        return DataSetConfigManager.get_effective_config(self.__class__, self._config)

    def set_instance_config(self, config: DataSetConfigType) -> None:
        self._config = config

    def set_instance_config_kwargs(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self._config, key, value)

    @classmethod
    def set_class_config(cls, config: DataSetConfigType) -> None:
        from mescal.data_sets.data_set_config import DataSetConfigManager
        DataSetConfigManager.set_class_config(cls, config)

    @classmethod
    def _get_class_name_lower_snake(cls) -> str:
        return to_lower_snake(cls.__name__)

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash((self.name, self._config))
