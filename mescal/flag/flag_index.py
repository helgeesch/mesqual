from __future__ import annotations
from typing import Set, Dict, TYPE_CHECKING, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps

from mescal.typevars import FlagType
from mescal.enums import ItemTypeEnum, VisualizationTypeEnum, TopologyTypeEnum, QuantityTypeEnum
from mescal.units import Units

if TYPE_CHECKING:
    from mescal.data_sets.data_set import DataSet


@dataclass
class RegistryEntry:
    flag: FlagType
    linked_model_flag: FlagType = None
    item_type: ItemTypeEnum = None
    visualization_type: VisualizationTypeEnum = None
    topology_type: TopologyTypeEnum = None
    unit: Units.Unit = None
    membership_column_name: str = None


def return_from_explicit_registry_if_available(attribute):
    def decorator(method):
        @wraps(method)
        def wrapper(self: FlagIndex, flag: FlagType, *args, **kwargs):
            if flag in self._explicit_registry:
                return getattr(self._explicit_registry[flag], attribute)
            return method(self, flag, *args, **kwargs)
        return wrapper
    return decorator


class FlagIndex(Generic[FlagType], ABC):
    def __init__(self, data_set: DataSet = None):
        self._explicit_registry: Dict[FlagType, RegistryEntry] = dict()
        self.linked_data_set = data_set

    def register_new_flag(
            self,
            flag: FlagType,
            linked_model_flag: FlagType = None,
            item_type: ItemTypeEnum = None,
            visualization_type: VisualizationTypeEnum = None,
            topology_type: TopologyTypeEnum = None,
            unit: Units.Unit = None,
    ):
        self._explicit_registry[flag] = RegistryEntry(
            flag,
            linked_model_flag,
            item_type,
            visualization_type,
            topology_type,
            unit,
        )

    def get_registry_entry(self, flag: FlagType) -> RegistryEntry:
        if flag in self._explicit_registry:
            return self._explicit_registry[flag]
        pseudo_entry = RegistryEntry(
            flag=flag,
            linked_model_flag=self.get_linked_model_flag(flag),
            item_type=self.get_item_type(flag),
            visualization_type=self.get_visualization_type(flag),
            topology_type=self.get_topology_type(flag),
            unit=self.get_unit(flag),
        )
        return pseudo_entry

    @return_from_explicit_registry_if_available('linked_model_flag')
    def get_linked_model_flag(self, flag: FlagType) -> FlagType:
        return self._get_linked_model_flag(flag)

    @return_from_explicit_registry_if_available('item_type')
    def get_item_type(self, flag: FlagType) -> ItemTypeEnum:
        return self._get_item_type(flag)

    @return_from_explicit_registry_if_available('visualization_type')
    def get_visualization_type(self, flag: FlagType) -> VisualizationTypeEnum:
        return self._get_visualization_type(flag)

    @return_from_explicit_registry_if_available('topology_type')
    def get_topology_type(self, flag: FlagType) -> TopologyTypeEnum:
        return self._get_topology_type(flag)

    @return_from_explicit_registry_if_available('unit')
    def get_unit(self, flag: FlagType) -> Units.Unit:
        return self._get_unit(flag)

    def get_quantity_type_enum(self, flag: FlagType) -> QuantityTypeEnum:
        unit = self.get_unit(flag)
        return Units.get_quantity_type_enum(unit)

    def get_all_timeseries_flags_for_model_flag(self, data_set: DataSet, flag: FlagType) -> Set[FlagType]:
        variable_flags = set()
        for f in data_set.accepted_flags:
            if self.get_item_type(f) == ItemTypeEnum.TimeSeries:
                if self.get_linked_model_flag(f) == flag:
                    variable_flags.add(f)
        return variable_flags

    @abstractmethod
    def get_flag_from_string(self, flag_string: str) -> FlagType:
        return flag_string

    @abstractmethod
    def _get_linked_model_flag(self, flag: FlagType) -> FlagType:
        raise NotImplementedError

    @abstractmethod
    def _get_item_type(self, flag: FlagType) -> ItemTypeEnum:
        raise NotImplementedError

    @abstractmethod
    def _get_visualization_type(self, flag: FlagType) -> VisualizationTypeEnum:
        raise NotImplementedError

    @abstractmethod
    def _get_topology_type(self, flag: FlagType) -> TopologyTypeEnum:
        raise NotImplementedError

    @abstractmethod
    def _get_unit(self, flag: FlagType) -> Units.Unit:
        raise NotImplementedError

    def get_linked_model_flag_for_membership_column(self, membership_column_name: str) -> FlagType:
        """
        In case the membership_column_name defines the membership towards an object (in another object class),
        this method returns the flag for the linked model.

        Example:
             The generator_model_df has a column 'node', which links to an object of the 'Node' type.
             The membership_coolumn_name 'node' will return 'Node.Model'.
        """
        for reg_entry in self._explicit_registry.values():
            if reg_entry.membership_column_name == membership_column_name:
                return reg_entry.flag
        return self._get_linked_model_flag_for_membership_column(membership_column_name)

    @abstractmethod
    def _get_linked_model_flag_for_membership_column(self, membership_column_name: str) -> FlagType:
        """
        In case the membership_column_name defines the membership towards an object (in another object class),
        this method returns the flag for the linked model.

        Example:
             The generator_model_df has a column 'node', which links to an object of the 'Node' type.
             The membership_coolumn_name 'node' will return 'Node.Model'.
        """
        if membership_column_name.lower() == 'node':
            return 'Node.Model'
        raise KeyError(f'No model linked to membership column {membership_column_name}.')

    @return_from_explicit_registry_if_available('membership_column_name')
    def get_membership_column_name_for_model_flag(self, flag: FlagType) -> str:
        """
        In case objects from other classes can have a membership with one of those objects (e.g. generator -> node),
        this method returns the name for the respective column in the model_df.

        Example:
            Objects that have a membership with a node (e.g. generators), have a column 'node' in the model_df.
            So flag 'Node.Model' will return 'node'.
        """
        if self.get_item_type(flag) != ItemTypeEnum.Model:
            raise ValueError('Method only valid for flags of type "Model".')
        return self._get_membership_column_name_for_model_flag(flag)

    @abstractmethod
    def _get_membership_column_name_for_model_flag(self, flag: FlagType) -> str:
        if flag == 'Node.Model':
            return 'node'
        raise KeyError(f'No membership column linked to flag {flag}.')

    def column_name_in_model_describes_membership(self, column_name: str) -> bool:
        try:
            _ = self.get_linked_model_flag_for_membership_column(column_name)
            return True
        except KeyError:
            return False

    @classmethod
    def get_flag_type(cls) -> type[FlagType]:
        from mescal.flag.flag import FlagTypeProtocol
        return FlagTypeProtocol


class EmptyFlagIndex(FlagIndex):
    def get_flag_from_string(self, flag_string: str) -> FlagType:
        return flag_string

    def _get_all_timeseries_flags_for_model_flag_from_implicit_registry(self, flag: FlagType) -> Set[FlagType]:
        return set()

    def _get_linked_model_flag(self, flag: FlagType) -> FlagType:
        raise NotImplementedError

    def _get_item_type(self, flag: FlagType) -> ItemTypeEnum:
        return ItemTypeEnum.Other

    def _get_visualization_type(self, flag: FlagType) -> VisualizationTypeEnum:
        return VisualizationTypeEnum.Other

    def _get_topology_type(self, flag: FlagType) -> TopologyTypeEnum:
        return TopologyTypeEnum.Other

    def _get_unit(self, flag: FlagType) -> Units.Unit:
        return Units.NaU

    def _get_linked_model_flag_for_membership_column(self, membership_column_name: str) -> FlagType:
        raise NotImplementedError

    def _get_membership_column_name_for_model_flag(self, flag: FlagType) -> str:
        raise NotImplementedError
