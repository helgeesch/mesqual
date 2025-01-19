from __future__ import annotations

from typing import Iterable, Iterator, Generic
from collections import Counter

import pandas as pd

from mescal.typevars import KPICollectionType
from mescal.kpis.kpi_base import KPI
from mescal.utils.pretty_scaling import get_pretty_min_max, get_pretty_order_of_mag, get_pretty_num_of_decimals
from mescal.utils.intersect_dicts import get_intersection_of_dicts
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class KPICollection(Generic[KPICollectionType]):
    def __init__(self, kpis: Iterable[KPI] = None):
        self._kpis: set[KPI] = set()
        if kpis:
            self.add_kpis(kpis)

    def add_kpis(self, kpis: Iterable[KPI]):
        for kpi in kpis:
            self.add_kpi(kpi)

    def add_kpi(self, kpi: KPI):
        if kpi in self._kpis:
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

    def get_kpi_series(self, pretty_text: bool = False, order_of_magnitude: float = None) -> pd.Series:
        # TODO: pretty_text
        # TODO: specify order_of_magnitude
        return pd.Series(
            {kpi.name: kpi.value for kpi in self._kpis}
        )

    def get_kpi_df(self, unstack_column_levels: str | list[str] = None) -> pd.DataFrame:
        df = pd.concat([kpi.get_kpi_as_series() for kpi in self._kpis], axis=1).T
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
        subset = self.get_kpi_group_by_attributes(**kwargs)
        num = len(subset._kpis)
        if num == 0:
            logger.warning(f'No KPI with matching attributes found.')
            return None
        elif num > 1:
            logger.warning(f'Found {num} KPIs for attributes {kwargs}. Returning the first one.')
        return next(iter(self._kpis))

    def get_kpi_group_by_attributes(self, **kwargs) -> 'KPIGroup':
        filtered_kpis = KPIGroup()
        for kpi in self._kpis:
            if kpi.has_attribute_values(**kwargs):
                filtered_kpis.add_kpi(kpi)
        return filtered_kpis

    def remove_kpi(self, kpi: KPI):
        self._kpis.remove(kpi)

    def remove_kpis_by_attributes(self, **kwargs) -> KPICollectionType:
        subset = self.get_kpi_group_by_attributes(**kwargs)
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


class KPIGroup(KPICollection['KPIGroup']):
    def __init__(self, kpis: Iterable[KPI] = None):
        super().__init__(kpis)

    @property
    def name(self) -> str:
        return ' '.join(self.get_in_common_kpi_attributes())

    def get_most_common_unit(self):
        unit_counts = Counter([kpi.unit for kpi in self._kpis])
        most_common = unit_counts.most_common(1)[0][0]
        if len(unit_counts) > 1:
            logger.warning(f'Multiple units identified in your KPI group. {most_common} is used.')
        return most_common

    def get_prettiest_order_of_magnitude_for_group(self):
        return get_pretty_order_of_mag(self.all_values)

    def get_prettiest_num_of_decimals_for_group(self):
        return get_pretty_num_of_decimals(self.all_values)

    def get_pretty_min_max_for_linear_scale(
            self,
            lower_percentile: int = 1,
            upper_percentile: int = 99,
    ) -> tuple[float, float]:
        return get_pretty_min_max(self.all_values, lower_percentile=lower_percentile, upper_percentile=upper_percentile)

    @property
    def all_values(self) -> list[bool | int | float]:
        return [kpi.value for kpi in self._kpis]

    def get_in_common_kpi_attributes(self) -> dict[str, bool | int | float | str]:
        dicts = [kpi.get_kpi_attributes_with_immutable_values() for kpi in self]
        return get_intersection_of_dicts(dicts)

    def get_group_without(self, kpi: KPI) -> 'KPIGroup':
        if kpi not in self._kpis:
            logger.warning(
                f'NotIntended: You are getting all kpis except a given one, '
                f'but the given one is not even in this {type(self)}. Sure this is intended?'
            )
        return KPIGroup({k for k in self._kpis if not k == kpi})
