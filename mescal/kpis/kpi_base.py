from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
import pandas as pd

from mescal.data_sets.data_set import DataSet
from mescal.data_sets.data_set_comparison import DataSetComparison
from mescal.typevars import Flagtype, DataSetType, ValueOperationType, KPIType
from mescal.units import Units
from mescal.kpis.aggs import (
    ValueComparison, ValueComparisons,
    ArithmeticValueOperation, ArithmeticValueOperations,
)


KPI_VALUE_TYPES = int | float | bool


def _to_primitive_types(any_dict: dict) -> dict:
    d = dict()
    for k, v in any_dict.items():
        if isinstance(v, (bool, int, float, str)):
            d[k] = v
        else:
            d[k] = str(v)
    return d


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
    def quantity(self) -> Units.Quantity:
        return self.value * self.unit

    @abstractmethod
    def compute(self):
        pass

    @property
    @abstractmethod
    def unit(self) -> Units.Unit:
        pass

    def get_attributed_object_name(self) -> str | int:
        """
        Only necessary in case one wants to be able to retrieve the attributed object_name for a KPI.
        For example you'd want this in case you need this info for plotting.
        """
        raise NotImplementedError

    def get_attributed_model_flag(self) -> Flagtype:
        """
        Only necessary in case one wants to be able to retrieve the attributed model_flag for a KPI.
        For example you'd want this in case you need this info for plotting.
        """
        raise NotImplementedError

    def get_attributed_object_info_from_model(self) -> pd.Series:
        model_flag = self.get_attributed_model_flag()
        model_df = self._data_set.fetch(model_flag)
        object_name = self.get_attributed_object_name()
        if object_name in model_df.index:
            return model_df.loc[object_name]
        else:
            raise KeyError(f"No info found for object '{object_name}' in model_df for flag '{model_flag}'.")

    def get_pretty_text_value(
            self,
            decimals: int = None,
            order_of_magnitude: int = None,
            include_unit: bool = True,
            always_include_sign: bool = None,
    ) -> str:
        raise NotImplementedError

    def get_kpi_name_with_data_set_name(self, data_set_name_as_suffix: bool = True) -> str:
        if data_set_name_as_suffix:
            return f'{self.name} {self._data_set.name}'
        return f'{self._data_set.name} {self.name}'

    def get_kpi_attributes(self) -> dict:
        atts = dict(
            kpi_name=self.name,
            data_set_name=self._data_set.name,
            unit=self.unit,
        )
        from mescal.kpis.kpis_from_aggregations import NoColumnDefinedException, MultipleColumnsInSubsetException
        try:
            atts['object_name'] = self.get_attributed_object_name()
        except (NotImplementedError, NoColumnDefinedException, MultipleColumnsInSubsetException):
            atts['object_name'] = None
        try:
            atts['model_flag'] = self.get_attributed_model_flag()
        except NotImplementedError:
            atts['model_flag'] = None
        return atts

    def get_kpi_attributes_as_hashable_values(self) -> dict:
        return _to_primitive_types(self.get_kpi_attributes())

    def get_kpi_as_series(self) -> pd.Series:
        s = self.get_kpi_attributes_as_hashable_values()
        s['value'] = self.value
        s['quantity'] = self.quantity
        return pd.Series(s, name=self.get_kpi_name_with_data_set_name())

    def has_attribute_values(self, **kwargs) -> bool:
        """Check if KPI matches all provided attribute conditions.

        Example:
            kpi = WhateverKPI(
                kpi_name='BiddingZone.MarketPrice',
                object_name="DE",
                aggregation='Mean',
                data_set_name='my_ds_1'
            )
            # Regular attribute checks
            kpi.has_attribute_values(name='BiddingZone.MarketPrice')  # True
            kpi.has_attribute_values(aggregation='Mean')  # True

            # None checks - all equivalent
            kpi.has_attribute_values(object_name_not_none=True)  # True
            kpi.has_attribute_values(object_name_not_na=True)    # True
            kpi.has_attribute_values(object_name_is_none=False)  # True
            kpi.has_attribute_values(object_name_isna=False)     # True

            # Multiple conditions
            kpi.has_attribute_values(
                name='BiddingZone.MarketPrice',
                data_set_name='my_ds_1'
            )  # True
        """
        if not kwargs:
            return True

        my_atts = self.get_kpi_attributes()
        my_atts_hashable = _to_primitive_types(my_atts)

        none_true_keys = ['_is_na', '_isna', '_is_nan', '_is_none']
        none_false_keys = ['_not_na', '_not_isna', '_not_nan', '_not_none']

        def _check_none_condition(kkey: str, value) -> bool:
            base_key = kkey
            for suffix in none_true_keys + none_false_keys:
                base_key = base_key.replace(suffix, '')

            attr_value = my_atts.get(base_key) or getattr(self, base_key, None) or getattr(self, f'_{base_key}', None)
            is_none_check = any(kkey.endswith(suffix) for suffix in none_true_keys)

            return (attr_value is None) == (is_none_check == value)

        for key, value in kwargs.items():
            if any(suffix in key for suffix in none_true_keys + none_false_keys):
                if not _check_none_condition(key, value):
                    return False
                continue
            elif key in my_atts and my_atts[key] == value:
                continue
            elif key in my_atts_hashable and my_atts_hashable[key] == value:
                continue
            elif hasattr(self, key) and getattr(self, key) == value:
                continue
            elif hasattr(self, f'_{key}') and getattr(self, f'_{key}') == value:
                continue
            else:
                return False

        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KPI):
            return False
        if self._data_set != other._data_set:
            return False
        return self.get_kpi_attributes_as_hashable_values() == other.get_kpi_attributes_as_hashable_values()

    def __hash__(self) -> int:
        imm = self.get_kpi_attributes_as_hashable_values()

        def _convert_dict_to_frozenset_of_tuples_for_hashability(d: dict):
            return hash(frozenset(d.items()))
        
        return _convert_dict_to_frozenset_of_tuples_for_hashability(imm)

    @classmethod
    def from_factory(cls, data_set: DataSetType) -> KPIType:
        class _Factory(KPIFactory):
            def get_kpi(self, data_set: DataSetType) -> KPIType:
                return cls(data_set)
        return _Factory().get_kpi(data_set)

    @classmethod
    def get_factory_instance(cls) -> KPIFactory:
        class FactoryClass(KPIFactory):
            def get_kpi(self, data_set: DataSetType) -> KPIType:
                return cls(data_set)
        return FactoryClass()


class _ValueOperationKPI(Generic[KPIType, ValueOperationType], KPI):

    def __init__(
            self,
            variation_kpi: KPIType,
            reference_kpi: KPIType,
            value_operation: ValueOperationType,
            data_set: DataSet = None
    ):
        data_set = data_set or variation_kpi._data_set
        super().__init__(data_set)
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
    def unit(self) -> Units.Unit:
        if self._value_operation.unit is not None:
            return self._value_operation.unit

        ref_kpi_unit = self._reference_kpi.unit
        var_kpi_unit = self._variation_kpi.unit
        val_op = self._value_operation

        _vcs = [ValueComparisons.Increase, ValueComparisons.Decrease, ValueComparisons.Delta, ValueComparisons.Diff]
        _aos = [ArithmeticValueOperations.Sum, ArithmeticValueOperations.Diff, ArithmeticValueOperations.Delta]
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

    def get_kpi_attributes(self) -> dict:
        from mescal.utils.intersect_dicts import get_intersection_of_dicts
        value_op_atts = dict()
        value_op_atts['variation_data_set'] = self._variation_kpi._data_set.name
        value_op_atts['reference_data_set'] = self._reference_kpi._data_set.name
        var_kpi_atts = self._variation_kpi.get_kpi_attributes_as_hashable_values()
        ref_kpi_atts = self._reference_kpi.get_kpi_attributes_as_hashable_values()
        var_ref_intersection = get_intersection_of_dicts([var_kpi_atts, ref_kpi_atts])
        value_op_atts.update(**var_ref_intersection)

        value_op_atts.update(value_operation=str(self._value_operation))

        return {**super().get_kpi_attributes(), **value_op_atts}


class ValueComparisonKPI(Generic[KPIType], _ValueOperationKPI[KPIType, ValueComparison]):

    def get_attributed_object_name(self) -> str | int:
        return self._variation_kpi.get_attributed_object_name()

    def get_attributed_model_flag(self) -> Flagtype:
        return self._variation_kpi.get_attributed_model_flag()

    def get_attributed_object_info_from_model(self) -> pd.Series:
        model_flag = self.get_attributed_model_flag()
        model_df = self._variation_kpi._data_set.fetch(model_flag)
        object_name = self.get_attributed_object_name()
        if object_name in model_df.index:
            return model_df.loc[object_name]
        else:
            raise KeyError(f"No info found for object '{object_name}' in model_df for flag '{model_flag}'.")



class ArithmeticValueOperationKPI(Generic[KPIType], _ValueOperationKPI[KPIType, ArithmeticValueOperation]):
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
            raise TypeError(f'Expected {DataSetComparison.__name__} for {self.__class__.__name__}, got {type(data_set)}.')

        var_kpi = self._kpi_factory.get_kpi(data_set.variation_data_set)
        ref_kpi = self._kpi_factory.get_kpi(data_set.reference_data_set)
        val_op = self._value_comparison
        val_comp_kpi = ValueComparisonKPI(var_kpi, ref_kpi, val_op, data_set)
        return val_comp_kpi


class ArithmeticOpKPIFactory(KPIFactory[DataSet, ArithmeticValueOperationKPI]):
    def __init__(
            self,
            var_kpi_factory: KPIFactory,
            ref_kpi_factory: KPIFactory,
            arithmetic_op: ArithmeticValueOperation,
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
    