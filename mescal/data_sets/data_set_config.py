from dataclasses import dataclass
from typing import Dict, Type, Optional, overload

from mescal.typevars import DataSetConfigType
from mescal.data_sets.data_set import DataSet


class InvalidConfigSettingError(Exception):
    pass


@dataclass
class DataSetConfig:
    use_database: bool = True
    auto_sort_datetime_index: bool = True
    remove_duplicate_indices: bool = True

    def merge(self, other: Optional[DataSetConfigType | dict]) -> DataSetConfigType:
        if other is None:
            return self

        merged_config = self.__class__()

        for attr_name in dir(self):
            if not attr_name.startswith('_'):  # Skip private attributes
                setattr(merged_config, attr_name, getattr(self, attr_name))

        if isinstance(other, dict):
            for key, value in other.items():
                if value is not None:
                    setattr(merged_config, key, value)
            return merged_config

        for attr_name in dir(other):
            if not attr_name.startswith('_'):
                other_value = getattr(other, attr_name)
                if other_value is not None:
                    setattr(merged_config, attr_name, other_value)

        return merged_config

    def __repr__(self) -> str:
        attrs = {
            name: getattr(self, name)
            for name in dir(self)
            if not name.startswith('_') and not callable(getattr(self, name))
        }
        return f"{self.__class__.__name__}({attrs})"


class DataSetConfigManager:
    _class_configs: Dict[Type[DataSet], DataSetConfig] = {}

    @classmethod
    @overload
    def set_class_config(cls, dataset_class: Type[DataSet], config: DataSetConfigType) -> None:
        ...

    @classmethod
    def set_class_config(cls, dataset_class: Type[DataSet], config: DataSetConfig) -> None:
        cls._class_configs[dataset_class] = config

    @classmethod
    @overload
    def update_class_config_kwargs(cls, dataset_class: Type[DataSet], **config_kwargs) -> None:
        ...

    @classmethod
    def update_class_config_kwargs(cls, dataset_class: Type[DataSet], **config_kwargs) -> None:
        for k, v in config_kwargs.items():
            setattr(cls._class_configs[dataset_class], k, v)

    @classmethod
    @overload
    def get_effective_config(
            cls,
            dataset_class: Type[DataSet],
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
