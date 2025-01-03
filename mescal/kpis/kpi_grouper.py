from typing import Iterable, Hashable
from operator import attrgetter
from collections import Counter
from itertools import groupby

from mescal.kpis.kpi_base import KPI, ValueComparisonKPI
from mescal.kpis.kpis_from_aggregations import FlagAggKPI
from mescal.kpis.kpi_collection import KPICollection
from mescal.utils.pretty_scaling import get_pretty_min_max, get_pretty_order_of_mag, get_pretty_num_of_decimals
from mescal.utils.intersect_dicts import get_intersection_of_dicts
from mescal.utils.logging import get_logger

logger = get_logger(__name__)

KPITypes = FlagAggKPI | ValueComparisonKPI[FlagAggKPI]


class KPIGroup(KPICollection):
    def __init__(self, kpis: Iterable[KPITypes] = None):
        super().__init__(kpis, include_data_set_name_in_naming=True)

    @property
    def name(self) -> str:
        return ' '.join(self.get_in_common_attribute_info())

    def get_unit(self):
        unit_counts = Counter([kpi.unit for kpi in self.kpi_iterator])
        most_common = unit_counts.most_common(1)[0][0]
        if len(unit_counts) > 1:
            logger.warning(f'Multiple units identified in your KPI group. {most_common} is used.')
        return most_common

    def get_prettiest_order_of_magnitude_for_group(self):
        return get_pretty_order_of_mag(self.all_values)

    def get_prettiest_num_of_decimals_for_group(self):
        return get_pretty_num_of_decimals(self.all_values)

    def get_pretty_min_max_for_linear_scale(self) -> tuple[float, float]:
        return get_pretty_min_max(self.all_values, lower_percentile=1, upper_percentile=99)

    @property
    def all_values(self) -> list[float | int | bool]:
        return [kpi.value for kpi in self.kpi_iterator]

    def get_in_common_attribute_info(self) -> dict[str, float | int | bool]:
        kpi_info_dicts = [kpi.get_kpi_info_as_dict() for kpi in self.kpi_iterator]
        return get_intersection_of_dicts(kpi_info_dicts)

    def get_all_kpis_except(self, kpi: KPITypes) -> set[KPITypes]:
        if kpi not in self.kpi_iterator:
            logger.warning(
                f'NotIntended: You are getting all kpis except a given one, '
                f'but the given one is not even in this {type(self)}. Sure this is intended?'
            )
        return {k for k in self.kpi_iterator if not k == kpi}


class KPIGrouper(KPICollection):
    def __init__(self, groupby_keys: list[str], kpis: Iterable[KPITypes]):
        super().__init__(kpis, include_data_set_name_in_naming=True)
        self.groupby_keys = groupby_keys
        self.groups: dict[Hashable, KPIGroup] = dict()
        self.perform_groupby()

    def perform_groupby(self):
        _attrgetter = attrgetter(*self.groupby_keys)
        kpis = list(self.kpi_iterator)
        kpis.sort(key=_attrgetter)
        for key, group in groupby(kpis, key=_attrgetter):
            # kwargs = {k: v for k, v in zip(self.groupby_keys, key)}
            self.groups[key] = KPIGroup(group)

    def get_group_key_for_kpi(self, kpi: KPITypes) -> tuple:
        return tuple(getattr(kpi, att) for att in self.groupby_keys)

    def get_group_for_kpi(self, kpi: KPITypes) -> KPIGroup:
        group_key = self.get_group_key_for_kpi(kpi)
        return self.groups[group_key]

    def get_all_other_kpis_from_group(self, kpi: KPITypes) -> set[KPITypes]:
        group = self.get_group_for_kpi(kpi)
        return group.get_all_kpis_except(kpi)

    @classmethod
    def factory_for_grouper_by_flag_and_agg(cls, kpis: Iterable[KPITypes]):
        """In order to iterate over all: datasets, column_subsets."""
        return cls(['flag', 'aggregation'], kpis)

    @classmethod
    def factory_for_grouper_by_dataset_flag_and_column_subset(cls, kpis: Iterable[KPITypes]):
        """In order to iterate over all: aggregations."""
        return cls(['data_set', 'flag', 'column_subset'], kpis)

    @classmethod
    def factory_for_grouper_by_flag_column_subset_and_agg(cls, kpis: Iterable[KPITypes]):
        """In order to iterate over all: datasets"""
        return cls(['flag', 'column_subset', 'aggregation'], kpis)


class KPIGroupers:
    def __init__(self):
        self._groupers: dict[tuple, KPIGrouper] = dict()

    def __getitem__(self, item: tuple):
        pass
