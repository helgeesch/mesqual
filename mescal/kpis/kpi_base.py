from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TYPE_CHECKING

import pandas as pd
import numpy as np

from mescal.data_sets.data_set import DataSet
from mescal.data_sets.data_set_comparison import DataSetComparison
from mescal.typevars import Flagtype, DataSetType, ValueOperationType, KPIType
from mescal import units
from mescal.kpis.aggs import (
    ValueComparison, ValueComparisons,
    ArithmeticOperation, ArithmeticOperations,
)


KPI_VALUE_TYPES = int | float | bool


class KPI(ABC):
    def __init__(self, data_set: DataSetType):
        self._data_set = data_set
        self._value: KPI_VALUE_TYPES = np.nan
        self._has_been_computed: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def required_flags(self) -> set[Flagtype]:
        pass

    @property
    def value(self) -> KPI_VALUE_TYPES:
        if not self._has_been_computed:
            self.compute()
        return self._value

    @value.setter
    def value(self, value: KPI_VALUE_TYPES):
        self._value = value

    @property
    def quantity(self) -> units.Quantity:
        return self.value * self.unit

    @abstractmethod
    def compute(self):
        pass

    @property
    @abstractmethod
    def unit(self) -> units.Unit:
        pass

    def get_pretty_text_value(
            self,
            decimals: int = None,
            order_of_magnitude: int = None,
            include_unit: bool = True
    ) -> str:
        # TODO 15
        return f'{self.value:.2f}'

    def __hash__(self) -> int:
        return hash(self.name)

    def get_kpi_name_with_data_set_name(self, data_set_name_as_suffix: bool = True) -> str:
        if data_set_name_as_suffix:
            return f'{self.name} {self._data_set.name}'
        return f'{self._data_set.name} {self.name}'

    def get_kpi_info_as_dict(self) -> dict:
        kpi_info = dict(
            data_set=self._data_set.name,
            name=self.name,
            unit=self.unit,
            value=self.value,
        )
        return kpi_info

    def get_kpi_info_as_series(self) -> pd.Series:
        kpi_info = dict()
        for k, v in self.get_kpi_info_as_dict():
            if any(isinstance(v, i) for i in [int, float, bool, str]):
                kpi_info[k] = v
            else:
                kpi_info[k] = str(v)
        return pd.Series(kpi_info, name=self.get_kpi_name_with_data_set_name())

    @classmethod
    def from_factory(cls, data_set: DataSetType) -> KPI:
        class _Factory(KPIFactory):
            def get_kpi(self, data_set: DataSetType) -> KPI:
                return cls(data_set)
        return _Factory().get_kpi(data_set)


class _ValueOperationKPI(Generic[KPIType, ValueOperationType], KPI):

    def __init__(self, variation_kpi: KPIType, reference_kpi: KPIType, value_operation: ValueOperationType):
        super().__init__(variation_kpi._data_set)
        self._variation_kpi = variation_kpi
        self._reference_kpi = reference_kpi
        self._value_operation = value_operation

    def compute(self):
        var_kpi_value = self._variation_kpi.value
        ref_kpi_value = self._reference_kpi.value
        value_op = self._value_operation
        self._value = value_op(var_kpi_value, ref_kpi_value)
        self._has_been_computed = True

    def get_kpi_name_with_data_set_name(self, data_set_name_as_suffix: bool = True) -> str:
        var_ds_name = self._variation_kpi._data_set.name
        ref_ds_name = self._reference_kpi._data_set.name
        ds_name = f'{var_ds_name} to {ref_ds_name}'
        if data_set_name_as_suffix:
            return f'{self.name} {ds_name}'
        return f'{ds_name} {self.name}'

    def required_flags(self) -> set[Flagtype]:
        return self._variation_kpi.required_flags().union(self._reference_kpi.required_flags())

    @property
    def unit(self) -> units.Unit:
        if self._value_operation.unit is not None:
            return self._value_operation.unit

        ref_kpi_unit = self._reference_kpi.unit
        var_kpi_unit = self._variation_kpi.unit
        val_op = self._value_operation

        _vcs = [ValueComparisons.Increase, ValueComparisons.Decrease, ValueComparisons.Delta, ValueComparisons.Diff]
        _aos = [ArithmeticOperations.Sum, ArithmeticOperations.Diff, ArithmeticOperations.Delta]
        if val_op in _vcs+_aos:
            if ref_kpi_unit == var_kpi_unit:
                return ref_kpi_unit
        return val_op(1*var_kpi_unit, 1*ref_kpi_unit).units

    @property
    def name(self) -> str:
        ref_kpi_name = self._reference_kpi.name
        var_kpi_name = self._variation_kpi.name

        name_of_operation = self._value_operation.name

        if ref_kpi_name == var_kpi_name:
            return f"{ref_kpi_name} {name_of_operation}"

        from mescal.utils.string_union import find_difference_and_join
        return find_difference_and_join(var_kpi_name, ref_kpi_name)

    def get_kpi_info_as_dict(self) -> dict:
        from mescal.utils.intersect_dicts import get_intersection_of_dicts
        kpi_info = dict()

        var_kpi_info = self._variation_kpi.get_kpi_info_as_dict()
        ref_kpi_info = self._reference_kpi.get_kpi_info_as_dict()
        var_ref_intersection = get_intersection_of_dicts([var_kpi_info, ref_kpi_info])
        kpi_info.update(**var_ref_intersection)

        kpi_info.update(value_operation=str(self._value_operation))
        kpi_info.update(**super().get_kpi_info_as_dict())

        return kpi_info


class ValueComparisonKPI(Generic[KPIType], _ValueOperationKPI[KPIType, ValueComparison]):
    pass


class ArithmeticValueOperationKPI(Generic[KPIType], _ValueOperationKPI[KPIType, ArithmeticOperation]):
    pass


class KPIFactory(Generic[DataSetType, KPIType], ABC):

    @abstractmethod
    def get_kpi(self, data_set: DataSetType) -> KPIType:
        pass

    def get_kpi_name(self):
        return self.get_kpi(None).name

    def __call__(self, data_set: DataSetType) -> KPIType:
        return self.get_kpi(data_set)

    def __hash__(self):
        return hash(self.get_kpi_name())


class ComparisonKPIFactory(KPIFactory[DataSetComparison, ValueComparisonKPI]):

    def __init__(
            self,
            kpi_factory: KPIFactory,
            value_comparison: ValueComparison,
    ):
        super().__init__()
        self._kpi_factory = kpi_factory
        self._value_comparison = value_comparison

    def get_kpi(self, data_set: DataSetComparison) -> ValueComparisonKPI:

        if not isinstance(data_set, DataSetComparison):
            raise TypeError(f'Expected DataSetComparison for {self.__class__.__name__}, got {type(data_set)}.')

        var_kpi = self._kpi_factory.get_kpi(data_set.variation_data_set)
        ref_kpi = self._kpi_factory.get_kpi(data_set.reference_data_set)
        val_op = self._value_comparison
        return ValueComparisonKPI(var_kpi, ref_kpi, val_op)


class ArithmeticOpKPIFactory(KPIFactory[DataSet, ArithmeticValueOperationKPI]):
    def __init__(
            self,
            var_kpi_factory: KPIFactory,
            ref_kpi_factory: KPIFactory,
            arithmetic_op: ArithmeticOperation,
    ):
        super().__init__()
        self._var_kpi_factory = var_kpi_factory
        self._ref_kpi_factory = ref_kpi_factory
        self._arithmetic_op = arithmetic_op

    def get_kpi(self, data_set: DataSetType) -> ArithmeticValueOperationKPI:
        var_kpi = self._var_kpi_factory.get_kpi(data_set.variation_data_set)
        ref_kpi = self._ref_kpi_factory.get_kpi(data_set.reference_data_set)
        val_op = self._arithmetic_op
        return ArithmeticValueOperationKPI(var_kpi, ref_kpi, val_op)
    