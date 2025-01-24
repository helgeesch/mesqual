from __future__ import annotations

from typing import Iterable, Iterator
from collections import Counter
from collections import defaultdict

import pandas as pd

from mescal.kpis.kpi_base import KPI
from mescal.utils.pretty_scaling import (
    get_pretty_min_max,
    get_pretty_order_of_mag,
    get_pretty_num_of_decimals,
    symmetric_scaling_around_0_seems_appropriate,
)
from mescal.utils.intersect_dicts import get_intersection_of_dicts
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class KPICollection:
    def __init__(self, kpis: Iterable[KPI] = None, name: str = None):
        self._kpis: set[KPI] = set()
        if kpis:
            self.add_kpis(kpis)
        self._name = name

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        return f'{self.__class__.__name__} for' + ' '.join(self.get_in_common_kpi_attributes())

    def add_kpis(self, kpis: Iterable[KPI]):
        for kpi in kpis:
            self.add_kpi(kpi)

    def add_kpi(self, kpi: KPI):
        if kpi in self:
            kpi_atts = kpi.get_kpi_attributes()
            logger.info(
                f"KPI with attributes \n"
                f"{kpi_atts}\n "
                f"already exists in the collection and is not overwritten. \n"
                f"If you wish to overwrite, remove the kpi instance first or use a new instance (with a different id)."
            )
            return
        self._kpis.add(kpi)

    def compute_all(self):
        for kpi in self._kpis:
            kpi.compute()

    def get_kpi_series(
            self,
            pretty_text: bool = False,
            decimals: int = None,
            order_of_magnitude: float = None,
            include_unit: bool = None,
            always_include_sign: bool = None,
    ) -> pd.Series:
        if not pretty_text:
            return pd.Series(
                {kpi.name: kpi.value for kpi in self._kpis}
            )
        return pd.Series(
            {kpi.name: kpi.get_pretty_text_value(decimals, order_of_magnitude, include_unit, always_include_sign) for kpi in self._kpis}
        )

    def get_kpi_df_with_descriptive_attributes(
            self,
            unstack_column_levels: str | list[str] = None,
            include_pretty_text_value: bool = False,
            pretty_text_decimals: int = None,
            pretty_text_order_of_magnitude: float = None,
            pretty_text_include_unit: bool = True,
            pretty_text_always_include_sign: bool = None,
    ) -> pd.DataFrame:
        df = pd.concat(
            [
                kpi.get_kpi_as_series(
                    include_pretty_text_value=include_pretty_text_value,
                    pretty_text_decimals=pretty_text_decimals,
                    pretty_text_order_of_magnitude=pretty_text_order_of_magnitude,
                    pretty_text_include_unit=pretty_text_include_unit,
                    pretty_text_always_include_sign=pretty_text_always_include_sign,
                )
                for kpi in self._kpis
            ],
            axis=1
        ).T
        if unstack_column_levels:
            if isinstance(unstack_column_levels, str):
                unstack_column_levels = [unstack_column_levels]
            if not all(c in df.columns for c in unstack_column_levels):
                raise KeyError(f'Some of the unstack_column_levels where not found in the kpi')
            index_cols = [c for c in df.columns if c not in unstack_column_levels+['value']]
            df = df.pivot_table(
                index=index_cols,
                columns=unstack_column_levels,
                values='value'
            )
        return df

    def get_kpi_by_attributes(self, **kwargs) -> KPI:
        subset = self.get_filtered_kpi_collection_by_attributes(**kwargs)
        num = len(subset._kpis)
        if num == 0:
            logger.warning(f'No KPI with matching attributes found.')
            return None
        elif num > 1:
            logger.warning(f'Found {num} KPIs for attributes {kwargs}. Returning the first one.')
        return next(iter(self._kpis))

    def get_filtered_kpi_collection_by_attributes(self, **kwargs) -> 'KPICollection':
        filtered_kpi_collection = KPICollection()
        for kpi in self._kpis:
            if kpi.has_attribute_values(**kwargs):
                filtered_kpi_collection.add_kpi(kpi)
        return filtered_kpi_collection

    def remove_kpi(self, kpi: KPI):
        self._kpis.remove(kpi)

    def remove_kpis_by_attributes(self, **kwargs) -> 'KPICollection':
        subset = self.get_filtered_kpi_collection_by_attributes(**kwargs)
        num = len(subset._kpis)
        for kpi in subset:
            self.remove_kpi(kpi)
        logger.info(f'Removed {num} KPIs from Collection.')
        return self

    def __contains__(self, kpi: KPI) -> bool:
        return kpi in self._kpis

    def __iter__(self) -> Iterator[KPI]:
        for kpi in self._kpis:
            yield kpi

    def __len__(self) -> int:
        return len(self._kpis)

    def get_this_groupbs_groupby_keys_and_values(self) -> dict[str, str | int]:
        keys_and_values = dict()
        self.get_in_common_kpi_attributes()
        in_common = ...
        different_keys_for = ...

    def get_most_common_unit(self):
        unit_counts = Counter([kpi.unit for kpi in self._kpis])
        most_common = unit_counts.most_common(1)[0][0]
        if len(unit_counts) > 1:
            logger.warning(f'Multiple units identified in your {self.__class__.__name__}. {most_common} is used.')
        return most_common

    def get_prettiest_order_of_magnitude_for_collection(self):
        return get_pretty_order_of_mag(self.all_values)

    def get_prettiest_num_of_decimals_for_collection(self):
        return get_pretty_num_of_decimals(self.all_values)

    def get_pretty_min_max_for_linear_scale(
            self,
            lower_percentile: int = 1,
            upper_percentile: int = 99,
            symmetric_scaling_around_0: bool = False
    ) -> tuple[float, float]:
        return get_pretty_min_max(self.all_values, lower_percentile, upper_percentile, symmetric_scaling_around_0)

    def get_symmetric_scaling_around_0_seems_appropriate(self) -> bool:
        return symmetric_scaling_around_0_seems_appropriate(self.all_values)

    @property
    def all_values(self) -> list[bool | int | float]:
        return [kpi.value for kpi in self._kpis]

    def get_in_common_kpi_attributes(self) -> dict[str, bool | int | float | str]:
        dicts = [kpi.get_kpi_attributes_as_hashable_values() for kpi in self]
        return get_intersection_of_dicts(dicts)

    def get_not_in_common_kpi_attributes_and_value_sets(self) -> dict[str, set[bool | int | float | str]]:
        dicts = [kpi.get_kpi_attributes_as_hashable_values() for kpi in self]
        in_common_keys = get_intersection_of_dicts(dicts)
        all_keys = set([k for d in dicts for k in d.keys()])
        not_in_common_keys = all_keys.difference(in_common_keys)
        values = defaultdict(set)
        for d in dicts:
            for k in not_in_common_keys:
                values[k].add(d[k])
        return values

    def get_group_without(self, kpi: KPI) -> 'KPICollection':
        if kpi not in self._kpis:
            logger.warning(
                f'NotIntended: You are getting all kpis except a given one, '
                f'but the given one is not even in this {type(self)}. Sure this is intended?'
            )
        return KPICollection({k for k in self._kpis if not k == kpi})

    @property
    def empty(self) -> bool:
        return len(self._kpis) == 0
