"""Small datasets for the dataset tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mesqual.datasets import Dataset, DatasetConfig, PlatformDataset, SourceInventory, config_field
from mesqual.flag import FrameSchema, Level, TimeSeriesFlagSpec
from mesqual.units import Units


def series_spec(flag: str, **fields) -> TimeSeriesFlagSpec:
    """A time series spec without model flag, for tests that need any spec."""
    fields.setdefault('unit', Units.MW)
    fields.setdefault('schema', FrameSchema(index=(Level('time', 'time'),)))
    return TimeSeriesFlagSpec(name=flag, description=f'Test flag {flag}.', model_flag=None, **fields)


@dataclass
class PlatformConfig(DatasetConfig):
    convert_periods: bool = config_field(True, 'Convert period numbers to timestamps.')


@dataclass
class StudyConfig(PlatformConfig):
    apply_correction: bool = config_field(False, 'Apply the study correction.')


class StaticDataset(Dataset):
    """Returns fixed frames and records the effective config of each `_fetch`.

    Each flag gets a time series spec, unless `specs` are given.
    """

    def __init__(self, data: dict, name: str = None, specs: tuple = None, **kwargs):
        self.flag_specs = tuple(specs) if specs is not None else tuple(series_spec(flag) for flag in data)
        super().__init__(name=name, **kwargs)
        self._data = data
        self.seen_configs: list[DatasetConfig] = []

    def _fetch(self, flag, effective_config, **kwargs):
        self.seen_configs.append(effective_config)
        return self._data[flag].copy()


class PlatformStaticDataset(StaticDataset):
    @classmethod
    def get_config_type(cls):
        return PlatformConfig


class StudyStaticDataset(PlatformStaticDataset):
    @classmethod
    def get_config_type(cls):
        return StudyConfig


class Interpreter(Dataset):
    """Base interpreter type of `Platform`. Subclasses set `FLAGS` and `_fetch`.

    A subclass that sets `FLAGS` and no `flag_specs` gets one time series spec per flag.
    """

    FLAGS: frozenset = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'FLAGS' in cls.__dict__ and 'flag_specs' not in cls.__dict__:
            cls.flag_specs = tuple(series_spec(flag) for flag in sorted(cls.FLAGS))

    def __init__(self, parent_dataset: Dataset = None, prices: pd.DataFrame = None):
        super().__init__(parent_dataset=parent_dataset)
        self.prices = prices

    def _fetch(self, flag, effective_config, **kwargs):
        raise NotImplementedError


class Platform(PlatformDataset):
    def build_source_inventory(self) -> SourceInventory:
        return SourceInventory(fingerprint=self.name)

    @classmethod
    def get_child_dataset_type(cls):
        return Interpreter


def hourly_frame(start: str = '2024-01-01', periods: int = 72, tz: str = None) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq='h', tz=tz, name='datetime')
    return pd.DataFrame({'a': range(periods), 'b': range(periods)}, index=index)
