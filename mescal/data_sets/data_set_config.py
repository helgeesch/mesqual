from dataclasses import dataclass, field
from typing import ClassVar, Dict, Type, Optional, Any, TYPE_CHECKING, overload

if TYPE_CHECKING:
    from mescal.data_sets.data_set import DataSet


ConfigType = TypeVar('ConfigType', bound='DataSetConfig')


@dataclass
class DataSetConfig:
    def merge(self: ConfigType, other: Optional[ConfigType]) -> ConfigType:
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


class ConfigManager:
    _class_configs: Dict[Type[DataSet], DataSetConfig] = {}

    @classmethod
    @overload
    def set_class_config(cls, dataset_class: Type[DataSet[ConfigType]], config: ConfigType) -> None:
        ...

    @classmethod
    def set_class_config(cls, dataset_class: Type[DataSet], config: DataSetConfig) -> None:
        cls._class_configs[dataset_class] = config

    @classmethod
    @overload
    def get_effective_config(
            cls,
            dataset_class: Type[DataSet[ConfigType]],
            instance_config: Optional[ConfigType] = None
    ) -> ConfigType:
        ...

    @classmethod
    def get_effective_config(
            cls,
            dataset_class: Type[DataSet],
            instance_config: Optional[DataSetConfig] = None
    ) -> DataSetConfig:
        config_type = dataset_class.config_type
        base_config = config_type()
        class_config = cls._class_configs.get(dataset_class)

        if class_config:
            base_config = base_config.merge(class_config)
        if instance_config:
            base_config = base_config.merge(instance_config)

        return base_config


class DataSet:
    def __init__(self, config: Optional[DataSetConfig] = None):
        self._config = config

    @property
    def config(self) -> DataSetConfig:
        return ConfigManager.get_effective_config(self.__class__, self._config)


class TimeSeriesDataSet(DataSet):
    def __init__(self, config: Optional[DataSetConfig] = None):
        super().__init__(config)

    def process_data(self, data: Any) -> Any:
        if self.config.use_datetime_index:
            # Process with datetime index
            pass
        else:
            # Process with enumerated index
            pass


if __name__ == '__main__':
    # Set a class-wide config for all TimeSeriesDataSet instances
    ts_class_config = DataSetConfig(
        use_datetime_index=False,
        aggregation_method='sum'
    )
    ConfigManager.set_class_config(TimeSeriesDataSet, ts_class_config)

    # Create instance with default class config
    ts1 = TimeSeriesDataSet()
    print(f"ts1 config: {ts1.config}")

    # Create instance with custom config that overrides class config
    ts2 = TimeSeriesDataSet(DataSetConfig(use_datetime_index=True))
    print(f"ts2 config: {ts2.config}")

    # Create regular DataSet instance (uses base defaults)
    ds = DataSet()
    print(f"ds config: {ds.config}")