from dataclasses import dataclass
from typing import Dict, Type, Optional, overload

from mescal.typevars import DataSetConfigType
from mescal.data_sets.data_set import DataSet


class InvalidConfigSettingError(Exception):
    pass


@dataclass
class DataSetConfig:
    auto_sort_datetime_index: bool = True

    def merge(self: DataSetConfigType, other: Optional[DataSetConfigType]) -> DataSetConfigType:
        if other is None:
            return self

        merged_config = self.__class__()
        for field in self.__dataclass_fields__:
            other_value = getattr(other, field, None)
            if other_value is not None:
                setattr(merged_config, field, other_value)
            else:
                setattr(merged_config, field, getattr(self, field))
        return merged_config


class DataSetConfigManager:
    _class_configs: Dict[Type[DataSet], DataSetConfig] = {}

    @classmethod
    @overload
    def set_class_config(cls, dataset_class: Type[DataSet[DataSetConfigType]], config: DataSetConfigType) -> None:
        ...

    @classmethod
    def set_class_config(cls, dataset_class: Type[DataSet], config: DataSetConfig) -> None:
        cls._class_configs[dataset_class] = config

    @classmethod
    @overload
    def get_effective_config(
            cls,
            dataset_class: Type[DataSet[DataSetConfigType]],
            instance_config: Optional[DataSetConfigType] = None
    ) -> DataSetConfigType:
        ...

    @classmethod
    def get_effective_config(
            cls,
            dataset_class: Type[DataSet],
            instance_config: Optional[DataSetConfig] = None
    ) -> DataSetConfig:
        config_type = dataset_class.get_config_type()
        base_config = config_type()
        class_config = cls._class_configs.get(dataset_class)

        if class_config:
            base_config = base_config.merge(class_config)
        if instance_config:
            base_config = base_config.merge(instance_config)

        return base_config
