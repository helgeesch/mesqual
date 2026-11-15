"""Dataset configs, their layers and the class config registry."""

from __future__ import annotations

import copy
import dataclasses
import inspect
from abc import ABCMeta
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Literal, Optional, Type

import pandas as pd

from mesqual.datasets.time_index_filter import TimeIndexFilter
from mesqual.typevars import DatasetConfigType

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset


class InvalidConfigSettingError(Exception):
    """Raised when a configuration value is incompatible with its config option."""

    pass


def config_field(
    default: Any = dataclasses.MISSING, description: str = '', merge_per_key: bool = False, **kwargs: Any
) -> Any:
    """Declare a config field with a description.

    The description goes into the field metadata. `describe_config()` reads
    it from there.

    Args:
        default: Default value of the field.
        description: One-line description of the field.
        merge_per_key: For a dict field: a layer adds its keys to the dict of
            the lower layers instead of replacing it.
        **kwargs: Further arguments for `dataclasses.field`, for example
            `default_factory`.

    Returns:
        A `dataclasses.field` with the description in its metadata.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class MyPlatformConfig(DatasetConfig):
        ...     convert_timestamps: bool = config_field(True, 'Convert period numbers to timestamps.')
    """
    metadata = {**kwargs.pop('metadata', {}), 'description': description, 'merge_per_key': merge_per_key}
    return dataclasses.field(default=default, metadata=metadata, **kwargs)


class _ClearType:
    """Type of the `DatasetConfig.CLEAR` marker. There is one instance."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self):
        return _ClearType, ()

    def __repr__(self) -> str:
        return 'DatasetConfig.CLEAR'


class _DatasetConfigMeta(ABCMeta):
    """Records the constructor arguments of a config as its explicit fields."""

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        init_names = [
            p.name
            for p in list(inspect.signature(cls.__init__).parameters.values())[1:]
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        explicit = frozenset(init_names[: len(args)]) | frozenset(kwargs)
        object.__setattr__(instance, '_explicit_fields', explicit)
        return instance


@dataclass
class DatasetConfig(metaclass=_DatasetConfigMeta):
    """Base config for the fetch behavior of a dataset.

    A platform or a study subclasses it as a dataclass and adds fields with
    `config_field`. A dataset declares its config class with
    `Dataset.get_config_type()`.

    Layers resolve in this order, later wins: config class defaults, class
    config of each class in the MRO (base first), instance config, fetch-time
    config. A layer overrides only its explicit fields: the constructor
    arguments, fields assigned after construction, or the keys of a dict.
    An explicit `None` overrides like any other value. Set a field to
    `DatasetConfig.CLEAR` to reset it to its class default and override the
    lower layers.

    Examples:
        >>> prices = dataset.fetch('Node.Results.price', config=dict(auto_sort_index=False))
    """

    CLEAR: ClassVar[_ClearType] = _ClearType()

    use_cache: bool = config_field(True, 'Read from and write to the cache of the dataset.')
    auto_sort_index: bool = config_field(True, 'Sort a result by its row index, of any kind, time or not.')
    keep_duplicate_indices: Literal['first', 'last', 'all'] = config_field(
        'first',
        "Rows to keep for duplicate index entries: 'first' or 'last' drops the other rows "
        "with a warning, 'all' keeps every row.",
    )
    time_index_filter: TimeIndexFilter | tuple[TimeIndexFilter, ...] | None = config_field(
        None,
        'Row filter on one level of the time index of time series flags, or several filters '
        'that combine with AND. `level=None` means the datetime level. A result without the '
        'level is returned unchanged, so model flags pass through. '
        'Applies only in the outermost fetch, after the cache.',
    )
    param_defaults: dict[str, Any] | None = config_field(
        None,
        'Default values of flag parameters, by parameter name. A value applies to every flag that '
        'declares a parameter with that name. The outermost fetch raises for a name in the instance or '
        'fetch-time config that no parameter of the dataset declares; a class config may also name a '
        'dataset parameter of any dataset class. Layers merge per key.',
        merge_per_key=True,
    )
    granularity: str | pd.Timedelta | None = config_field(
        None,
        "Target time resolution of time series flags, a pandas frequency such as '15min' or '1h'. "
        '`None`: the native resolution. Applies only in the outermost fetch, after the cache.',
        kw_only=True,
    )

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if name[0] != '_' or name == '__dict__':
            recorder = object.__getattribute__(self, '__dict__').get('_read_recorder')
            if recorder is not None:
                self._record_read(recorder, name, value)
        return value

    @property
    def _state(self) -> dict:
        """The instance dict, read without recording."""
        return object.__getattribute__(self, '__dict__')

    def _record_read(self, recorder, name: str, value: Any) -> None:
        if name == '__dict__':
            for field_name in self._field_names():
                recorder(field_name, object.__getattribute__(self, field_name))
        elif name in type(self).__dataclass_fields__:
            recorder(name, value)

    def _watched(self, recorder) -> DatasetConfig:
        """Return a copy that reports each field read to `recorder(name, value)`."""
        watched = copy.copy(self)
        watched._state['_read_recorder'] = recorder
        return watched

    def __getstate__(self) -> dict:
        state = dict(self._state)
        recorder = state.pop('_read_recorder', None)
        if recorder is not None:
            self._record_read(recorder, '__dict__', None)
        return state

    def __setattr__(self, name: str, value: Any) -> None:
        if '_read_recorder' in self._state:
            raise AttributeError(
                f'The effective config of a fetch is read-only; cannot set {name!r}. '
                f'To pass a changed config on, use config.merge({{{name!r}: ...}}).'
            )
        super().__setattr__(name, value)
        if not name.startswith('_') and '_explicit_fields' in self._state:
            super().__setattr__('_explicit_fields', self._explicit_fields | {name})
            super().__setattr__('_cleared_fields', self._cleared_fields - {name})

    @property
    def _cleared_fields(self) -> frozenset[str]:
        return self._state.get('_cleared_fields_set', frozenset())

    @_cleared_fields.setter
    def _cleared_fields(self, names: frozenset[str]) -> None:
        self._state['_cleared_fields_set'] = names

    @property
    def explicit_fields(self) -> frozenset[str]:
        """Names of the fields that this config sets explicitly."""
        return self._state.get('_explicit_fields', frozenset())

    @classmethod
    def _merges_per_key(cls, name: str) -> bool:
        field = cls.__dataclass_fields__.get(name)
        return field is not None and field.metadata.get('merge_per_key', False)

    def _unknown_fields(self) -> frozenset[str]:
        return self.explicit_fields - self._field_names()

    @classmethod
    def _field_names(cls) -> frozenset[str]:
        return frozenset(f.name for f in dataclasses.fields(cls))

    @classmethod
    def _field_default(cls, name: str) -> Any:
        field = {f.name: f for f in dataclasses.fields(cls)}[name]
        if field.default_factory is not dataclasses.MISSING:
            return field.default_factory()
        return field.default

    def merge(self, other: Optional[DatasetConfigType | dict]) -> DatasetConfigType:
        """Return a new config with the explicit fields of `other` on top of this one.

        The result has the type of this config. Its explicit fields are the
        union of the explicit fields of both layers. An explicit `None` in
        `other` overrides like any other value. A field declared with
        `merge_per_key` (`param_defaults`) adds the keys of `other` to its
        dict. `CLEAR` resets a field to the
        class default of this config; the result remembers the reset, so a
        collection passes `CLEAR` on and each child resolves it against its
        own default. A name that this config class does not define is kept
        as is, so a collection can pass it on to its children.

        Args:
            other: Upper layer: `None`, a dict, or a config of any type.

        Returns:
            The merged config.

        Examples:
            >>> base = DatasetConfig(use_cache=False)
            >>> base.merge({'auto_sort_index': False}).use_cache
            False
        """
        merged = copy.copy(self)
        for name, value in self._explicit_items(other):
            if value is DatasetConfig.CLEAR and name in self._field_names():
                setattr(merged, name, self._field_default(name))
                merged._cleared_fields = merged._cleared_fields | {name}
            elif self._merges_per_key(name) and isinstance(value, dict):
                setattr(merged, name, {**(getattr(merged, name) or {}), **value})
            else:
                setattr(merged, name, value)
        return merged

    @staticmethod
    def _explicit_items(other: Optional[DatasetConfig | dict]) -> Iterable[tuple[str, Any]]:
        if other is None:
            return []
        if isinstance(other, dict):
            return other.items()
        if isinstance(other, DatasetConfig):
            return [
                (name, DatasetConfig.CLEAR if name in other._cleared_fields else getattr(other, name))
                for name in sorted(other.explicit_fields)
            ]
        raise TypeError(f'Config must be a dict or a {DatasetConfig.__name__}, got {type(other)}.')

    def __repr__(self) -> str:
        recorder = self._state.get('_read_recorder')
        if recorder is not None:
            self._record_read(recorder, '__dict__', None)
        attrs = {k: v for k, v in self._state.items() if not k.startswith('_')}
        return f'{self.__class__.__name__}({attrs})'


class DatasetConfigManager:
    """Registry of class configs, the class layer of the config hierarchy.

    Class configs are global state. A class config applies to all instances of
    that dataset class and of its subclasses.

    Examples:
        >>> from mesqual.testing import DummyDataset
        >>> DatasetConfigManager.set_class_config(DummyDataset, DatasetConfig(use_cache=False))
        >>> DatasetConfigManager.update_class_config_kwargs(DummyDataset, auto_sort_index=False)
        >>> DatasetConfigManager.clear_class_configs()
    """

    _class_configs: dict[Type[Dataset], DatasetConfig] = {}

    @classmethod
    def set_class_config(cls, dataset_class: Type[Dataset], config: DatasetConfig) -> None:
        """Set the class config of a dataset class.

        Args:
            dataset_class: Dataset class to configure.
            config: Config whose explicit fields override the config defaults.
        """
        cls._class_configs[dataset_class] = config

    @classmethod
    def update_class_config_kwargs(cls, dataset_class: Type[Dataset], **config_kwargs: Any) -> None:
        """Set fields on the existing class config of a dataset class.

        Args:
            dataset_class: Dataset class whose class config to update.
            **config_kwargs: Field names and values.

        Raises:
            KeyError: The dataset class has no class config.
        """
        if dataset_class not in cls._class_configs:
            raise KeyError(f'{dataset_class.__name__} has no class config. Call set_class_config first.')
        for k, v in config_kwargs.items():
            setattr(cls._class_configs[dataset_class], k, v)

    @classmethod
    def clear_class_configs(cls) -> None:
        """Remove all class configs."""
        cls._class_configs.clear()

    @classmethod
    def get_layers(
        cls,
        dataset_class: Type[Dataset],
        instance_config: Optional[DatasetConfig] = None,
    ) -> list[tuple[str, DatasetConfig]]:
        """Return the config layers of a dataset class, lowest first.

        Args:
            dataset_class: Dataset class.
            instance_config: Optional instance config, the top layer.

        Returns:
            Pairs of layer label and config: `'default'`, `'class <Name>'` for
                each class config along the MRO (base first), and `'instance'`.
        """
        layers = [('default', dataset_class.get_config_type()())]
        for klass in reversed(dataset_class.__mro__):
            class_config = cls._class_configs.get(klass)
            if class_config is not None:
                layers.append((f'class {klass.__name__}', class_config))
        if instance_config is not None:
            layers.append(('instance', instance_config))
        return layers

    @classmethod
    def get_effective_config(
        cls,
        dataset_class: Type[Dataset],
        instance_config: Optional[DatasetConfig] = None,
    ) -> DatasetConfig:
        """Resolve the config of a dataset class without the fetch-time layer.

        Args:
            dataset_class: Dataset class.
            instance_config: Optional instance config.

        Returns:
            The merged config, of the config type of the dataset class.
        """
        layers = cls.get_layers(dataset_class, instance_config)
        effective = layers[0][1]
        for _, config in layers[1:]:
            effective = effective.merge(config)
        return effective
