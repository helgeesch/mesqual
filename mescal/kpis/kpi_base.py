from __future__ import annotations

from typing import Generic, Hashable, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from mescal.data_sets.data_set import DataSet
from mescal.data_sets.data_set_comparison import DataSetComparison
from mescal.typevars import Flagtype, DataSetType, ValueOperationType, KPIType
from mescal.units import Units
from mescal.kpis.aggs import (
    ValueComparison, ValueComparisons,
    ArithmeticValueOperation, ArithmeticValueOperations,
    Aggregation
)


KPI_VALUE_TYPES = int | float | bool


@dataclass
class KPIAttributes:
    name: Optional[str] = None
    data_set: Optional[DataSet] = None
    data_set_type: Optional[type[DataSet]] = None
    unit: Optional[Units.Unit] = None
    base_unit: Optional[Units.Unit] = None
    flag: Optional[Flagtype] = None
    object_name: Optional[int | str] = None
    model_flag: Optional[Flagtype] = None
    aggregation: Optional[Aggregation] = None
    variation_data_set: Optional[DataSet] = None
    reference_data_set: Optional[DataSet] = None
    model_query: Optional[str] = None
    column_subset: Optional[Hashable | list[Hashable]] = None
    name_prefix: Optional[str] = None
    name_suffix: Optional[str] = None
    value_comparison: Optional[ValueComparison] = None
    value_operation: Optional[ArithmeticValueOperation] = None

    def as_dict(self, primitive_values: bool = False, include_none_values: bool = False) -> dict:
        if not primitive_values:
            return {k: v for k, v in self.__dict__.items() if include_none_values or v is not None}
        else:
            return {k: self._to_primitive(v) for k, v in self.as_dict(include_none_values).items()}

    def has_attr(self, attr_query: str = None, **kwargs) -> bool:
        primitive_dict = self.as_dict(primitive_values=True, include_none_values=True)
        df = pd.DataFrame([primitive_dict])

        if attr_query:
            try:
                df = df.query(attr_query, engine='python')
                if df.empty:
                    return False
            except pd.errors.UndefinedVariableError:
                return False

        return all((getattr(self, k, None) == v) or (primitive_dict.get(k, None) == v) for k, v in kwargs.items())

    def intersection(self, other: 'KPIAttributes') -> 'KPIAttributes':
        return KPIAttributes(**{k: v for k, v in self.__dict__.items() if k in other.__dict__ and other.__dict__[k] == v})

    @staticmethod
    def _to_primitive(value):
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, DataSet):
            return value.name
        return str(value)

    def update(self, other: 'KPIAttributes') -> 'KPIAttributes':
        for k, v in other.as_dict(include_none_values=False).items():
            setattr(self, k, v)
        return self


class KPI(ABC):
    def __init__(self, data_set: DataSetType):
        self._data_set = data_set
        self._value: KPI_VALUE_TYPES = np.nan
        self._has_been_computed: bool = False
        self._attributes: KPIAttributes = None
        self._post_init()

    def _post_init(self):
        pass

    @property
    def attributes(self) -> KPIAttributes:
        if self._attributes is None:
            self._attributes = self._get_kpi_attributes()
        return self._attributes

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

    def _get_kpi_attributes(self) -> KPIAttributes:
        atts = KPIAttributes(
            name=self.name,
            data_set=self._data_set,
            data_set_type=type(self._data_set),
            unit=self.unit,
            base_unit=Units.get_base_unit_for_unit(self.unit),
        )
        return atts

    def get_kpi_as_series(self) -> pd.Series:
        s = self.attributes.as_dict(primitive_values=True)
        s['value'] = self.value
        s['quantity'] = self.quantity
        return pd.Series(s, name=self.get_kpi_name_with_data_set_name())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KPI):
            return False
        if self._data_set != other._data_set:
            return False
        return self.attributes == other.attributes

    def __hash__(self) -> int:
        imm = self.attributes.as_dict(primitive_values=True, include_none_values=False)

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

    def _get_kpi_attributes(self) -> KPIAttributes:

        var_atts = self._variation_kpi.attributes
        ref_atts = self._reference_kpi.attributes

        this_atts: KPIAttributes = var_atts.intersection(ref_atts)

        this_atts.update(super()._get_kpi_attributes())
        this_atts.variation_data_set = self._variation_kpi._data_set
        this_atts.reference_data_set = self._reference_kpi._data_set
        this_atts.value_operation = self._value_operation
        return this_atts


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
    