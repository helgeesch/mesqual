from __future__ import annotations

from typing import Hashable, Generic, TYPE_CHECKING

import pandas as pd

from mescal import units
from mescal.typevars import DataSetType, Flagtype
from mescal.utils.pandas_utils.filter import filter_by_model_query
from mescal.kpis.kpi_base import KPI, KPIFactory
from mescal.utils.logging import get_logger

if TYPE_CHECKING:
    from mescal.kpis.aggs import Aggregation

logger = get_logger(__name__)

SPACE = ' '


class MultipleColumnsInSubset(Exception):
    pass


class NoColumnDefined(Exception):
    pass


class MissingColumnsFromSubset(Exception):
    pass


class FlagAggKPI(Generic[DataSetType], KPI):

    def __init__(
            self,
            data_set: DataSetType,
            flag: Flagtype,
            aggregation: Aggregation,
            column_subset: Hashable | list[Hashable] = None,
            model_query: str = None,
            kpi_name_prefix: str = None,
            kpi_name_suffix: str = None,
            kpi_name: str = None,
    ):
        super().__init__(data_set=data_set)

        self._flag = flag
        self._aggregation = aggregation
        self._column_subset = column_subset
        self._model_query = model_query
        self._kpi_name_prefix = kpi_name_prefix
        self._kpi_name_suffix = kpi_name_suffix
        self._kpi_name = kpi_name

    def get_kpi_attributes(self) -> dict:
        flag_agg_atts = dict(
            flag=self._flag,
            aggregation=self._aggregation,
            column_subset=self._column_subset,
            model_query=self._model_query,
            name_prefix=self._kpi_name_prefix,
            name_suffix=self._kpi_name_suffix,
        )
        return {**super().get_kpi_attributes(), **flag_agg_atts}

    @property
    def name(self) -> str:
        if self._kpi_name:
            return self._kpi_name
        return self._generic_kpi_name_generator(
            aggregation=self._aggregation,
            flag=self._flag,
            column_subset=self._column_subset,
            model_query=self._model_query,
            kpi_name_prefix=self._kpi_name_prefix,
            kpi_name_suffix=self._kpi_name_suffix,
        )

    @staticmethod
    def _generic_kpi_name_generator(
            flag: Flagtype,
            aggregation: Aggregation,
            column_subset: Hashable | list[Hashable] = None,
            model_query: str = None,
            kpi_name_prefix: str = None,
            kpi_name_suffix: str = None,
    ) -> str:
        name = ''

        if kpi_name_prefix is not None:
            name += kpi_name_prefix + SPACE

        name += f'{aggregation} {flag}'

        if model_query is not None:
            name += f' ("{model_query}")'
        if column_subset is not None:
            name += f' ({column_subset})'
        if kpi_name_suffix is not None:
            name += SPACE + kpi_name_suffix

        return name

    @property
    def unit(self) -> units.Unit:
        if self._aggregation.unit is not None:
            return self._aggregation.unit
        return self._data_set.flag_index.get_unit(self._flag)

    def _fetch_filtered_data(self, data_set: DataSetType) -> pd.DataFrame:
        flag = self._flag
        data = data_set.fetch(flag)

        if self._model_query:
            model_flag = data_set.flag_index.get_linked_model_flag(flag)
            model_df = data_set.fetch(model_flag)
            data = filter_by_model_query(data, model_df, self._model_query)

        if self._column_subset:
            subset = self._get_column_subset_as_list()
            if not set(subset).issubset(data.columns):
                missing = set(subset).difference(data.columns)
                raise MissingColumnsFromSubset(
                    f'Trying to fetch the data for {self.get_kpi_name_with_data_set_name()}, '
                    f'the following columns were not found in the result_df: {missing}.'
                )
            data = data[subset]

        return data

    def _get_column_subset_as_list(self) -> list[str | int]:
        return self._column_subset if isinstance(self._column_subset, list) else [self._column_subset]

    def get_attributed_object_name(self) -> str | int:
        """
        Only necessary in case one wants to be able to retrieve the attributed object_name for a KPI.
        For example you'd want this in case you need this info for plotting.
        """
        if self._column_subset is None:
            raise NoColumnDefined(
                f"You are trying to get the attributed_object_name for KPI {self.get_kpi_name_with_data_set_name()}. "
                f"However, this method is only valid if you define exactly 1 column in the column_subset. "
                f"Currently, the column_subset is not set at all (None)."
            )

        subset = self._get_column_subset_as_list()
        if len(subset) > 1:
            raise MultipleColumnsInSubset(
                f"You are trying to get the attributed_object_name for KPI {self.get_kpi_name_with_data_set_name()}. "
                f"However, This method is only valid if you define exactly 1 column in the column_subset. "
                f"Currently, the column_subset contains multiple columns: {self._column_subset}."
            )

        column = subset[0]
        if isinstance(column, tuple):
            model_flag = self.get_attributed_model_flag()
            model_df = self._data_set.fetch(model_flag)
            options = set(column).intersection(model_df.index)
            if len(options) == 1:
                return list(options)[0]
            elif len(options) > 1:
                selected_object = [i for i in column if i in options][0]
                logger.warning(
                    f"Column Subset appears to be a tuple ({column}) with multiple possible primary objects: {options}."
                    f"We are proceeding with the first one ({selected_object})"
                )
                return selected_object
        else:
            return column

    def get_attributed_model_flag(self) -> Flagtype:
        """
        Only necessary in case one wants to be able to retrieve the attributed model_flag for a KPI.
        For example you'd want this in case you need this info for plotting.
        """
        model_flag = self._data_set.flag_index.get_linked_model_flag(self._flag)
        return model_flag

    def compute(self):
        data = self._fetch_filtered_data(self._data_set)
        self._value = self._aggregation(data)
        self._has_been_computed = True

    def required_flags(self) -> set[Flagtype]:
        flags = {self._flag}
        if self._model_query:
            flags.add(self._get_model_flag_for_flag())
        return flags

    def _get_model_flag_for_flag(self) -> Flagtype:
        return self._data_set.flag_index.get_linked_model_flag(self._flag)


class FlagAggKPIFactory(Generic[DataSetType], KPIFactory[DataSetType, FlagAggKPI]):
    def __init__(
            self,
            flag: Flagtype,
            aggregation: Aggregation,
            column_subset: Hashable | list[Hashable] = None,
            model_query: str = None,
            kpi_name_prefix: str = None,
            kpi_name_suffix: str = None,
            kpi_name: str = None,
    ):
        self._flag = flag
        self._aggregation = aggregation
        self._column_subset = column_subset
        self._model_query = model_query
        self._kpi_name_prefix = kpi_name_prefix
        self._kpi_name_suffix = kpi_name_suffix
        self._kpi_name = kpi_name

    def get_kpi(self, data_set: DataSetType) -> FlagAggKPI:
        kpi = FlagAggKPI(
            data_set=data_set,
            flag=self._flag,
            aggregation=self._aggregation,
            column_subset=self._column_subset,
            model_query=self._model_query,
            kpi_name_prefix=self._kpi_name_prefix,
            kpi_name_suffix=self._kpi_name_suffix,
            kpi_name=self._kpi_name,
        )
        return kpi
