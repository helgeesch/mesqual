from __future__ import annotations

from typing import Dict, Iterable, List, Iterator

import pandas as pd

from mescal.kpis.kpi_base import KPI, KPI_VALUE_TYPES
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class KPICollection:
    def __init__(self, kpis: Iterable[KPI] = None, include_data_set_name_in_naming: bool = False):
        self._kpis: Dict[str, KPI] = {}
        self._include_data_set_name_in_naming = include_data_set_name_in_naming
        if kpis:
            self.add_kpis(kpis)

    @property
    def kpi_iterator(self) -> Iterator[KPI]:
        for kpi in self.get_all_kpis():
            yield kpi

    def add_kpis(self, kpis: Iterable[KPI]):
        for kpi in kpis:
            self.add_kpi(kpi)

    def _kpi_name(self, kpi: KPI) -> str:
        if self._include_data_set_name_in_naming:
            return kpi.get_kpi_name_with_data_set_name()
        return kpi.name

    def add_kpi(self, kpi: KPI):
        kpi_name = self._kpi_name(kpi)
        if kpi_name in self._kpis:
            existing = self._kpis[kpi_name]
            if id(kpi) == id(existing):
                logger.info(
                    f"KPI with name '{kpi_name}' already exists in the collection and "
                    f"is not overwritten because they seem to be the same object."
                    f"If you wish to overwrite, remove the kpi instance first or use a new instance (with a different id)."
                )
                return
            else:
                logger.info(
                    f"KPI with name '{kpi_name}' already existed in the collection and "
                    f"is now overwritten with the new instance, because the new kpi has a different id."
                )
        self._kpis[kpi_name] = kpi

    def get_kpi_series(self, **kwargs) -> pd.Series:
        # TODO: pretty_text
        # TODO: specify order_of_magnitude
        # TODO: subset
        return pd.Series(
            {self._kpi_name(kpi): kpi.value for kpi in self._kpis.values()}
        )

    def calculate_all(self):
        for kpi in self._kpis.values():
            kpi.calculate()

    def get_kpi_value(self, name: str) -> KPI_VALUE_TYPES:
        if name not in self._kpis:
            raise KeyError(f"No KPI found with name '{name}'.")
        return self._kpis[name].value

    def get_kpi(self, name: str) -> KPI:
        if name not in self._kpis:
            raise KeyError(f"No KPI found with name '{name}'.")
        return self._kpis[name]

    def remove_kpi(self, name: str):
        if name not in self._kpis:
            raise KeyError(f"No KPI found with name '{name}'.")
        del self._kpis[name]

    def get_all_kpis(self) -> Iterable[KPI]:
        return self._kpis.values()

    def get_all_kpis_with(self, **kwargs) -> List[KPI]:
        filtered_kpis = []
        for kpi in self._kpis.values():
            if all(
                getattr(kpi, k, getattr(kpi, f'_{k}', None)) == v
                for k, v in kwargs.items()
            ):
                filtered_kpis.append(kpi)
        return filtered_kpis

    def get_all_kpi_names(self) -> Iterable[str]:
        return self._kpis.keys()

    def __getitem__(self, item) -> KPI:
        return self._kpis[item]
