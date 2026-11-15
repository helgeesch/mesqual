"""Internal check of `Dataset.fetch` that decides if a stored fetch record is still up to date; not public API."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from mesqual.datasets.fetch_context import FetchRecord, FileRead

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset
    from mesqual.datasets.dataset_config import DatasetConfig


class DependencyCheck:
    """Decides if a stored fetch record is up to date for a fetch of `dataset` with `effective_config`.

    A record is up to date if its call has not raised, the version identity
    and the resolved parameters are the same, each recorded config read has
    the same value under the current config, each recorded file is unchanged
    (`FileRead.is_unchanged`), and each dependency is up to date. "The same"
    is the rule of `_same`. For a dependency, the check rebuilds the call:
    the current config of the dataset, then the recorded `call_config` (or
    the current config of the parent record for `INHERIT`), then the
    recorded `call_kwargs`, and compares the same items. The cache key holds
    the parameters of the top-level call only in canonical form, so the
    parameter comparison also catches a value with the same canonical form.

    The check finds the dataset of a dependency by its class and name in the
    dataset tree of `dataset` (from the root through all children). A
    dependency whose dataset it does not find is stale.

    Args:
        dataset: The dataset of the fetch.
        record: The record of the current call, before its fetch.
        effective_config: The config that `_fetch` of the current call gets.
    """

    def __init__(self, dataset: Dataset, record: FetchRecord, effective_config: DatasetConfig):
        self._dataset = dataset
        self._record = record
        self._config = effective_config
        self._datasets: dict[tuple[str, str], Dataset] | None = None
        self._unchanged: dict[FileRead, bool] = {}

    def stale_reason(self, stored: FetchRecord) -> str | None:
        """The first reason why `stored`, the record of an earlier fetch with the same cache key, is not up to date.

        For example `'config field x changed'`. `None` if it is up to date.
        """
        return self._stale_reason(stored, self._record, self._config)

    def _stale_reason(self, stored: FetchRecord, current: FetchRecord, config: DatasetConfig) -> str | None:
        if stored.raised:
            return 'the fetch raised'
        if not self._same(current.version, stored.version):
            return 'the version identity changed'
        if not self._same(current.params, stored.params):
            return 'the parameters changed'
        if stored.reads_given_params and not self._same(current.given_params, stored.given_params):
            return 'the given parameters changed'
        for name, value in stored.config_reads.items():
            if not self._same(getattr(config, name, dataclasses.MISSING), value):
                return f'config field {name} ' + ('changed' if self._same(value, value) else 'cannot be compared')
        for file_read in stored.file_reads:
            if not self._is_unchanged(file_read):
                return f'file {file_read.path} changed'
        for dependency in stored.dependencies:
            reason = self._dependency_stale_reason(dependency, config)
            if reason is not None:
                return f'{dependency.dataset_name} {dependency.flag}: {reason}'
        return None

    def _is_unchanged(self, file_read: FileRead) -> bool:
        """`FileRead.is_unchanged`, computed once per file state for this check."""
        if file_read not in self._unchanged:
            self._unchanged[file_read] = file_read.is_unchanged()
        return self._unchanged[file_read]

    def _dependency_stale_reason(self, stored: FetchRecord, parent_config: DatasetConfig) -> str | None:
        dataset = self._find(stored.dataset_class, stored.dataset_name)
        if dataset is None:
            return 'the dataset is not in the dataset tree'
        try:
            current, config = dataset._rebuild_call(stored, parent_config)
        except Exception as error:
            return f'the call cannot be rebuilt ({type(error).__name__})'
        return self._stale_reason(stored, current, config)

    def _find(self, dataset_class: str, dataset_name: str) -> Dataset | None:
        if self._datasets is None:
            self._datasets = {}
            seen: set[int] = set()
            pending = [self._dataset._root_dataset()]
            while pending:
                dataset = pending.pop(0)
                if id(dataset) in seen:
                    continue
                seen.add(id(dataset))
                self._datasets.setdefault((dataset._class_path(), dataset.name), dataset)
                pending.extend(dataset._child_datasets())
        return self._datasets.get((dataset_class, dataset_name))

    @classmethod
    def _same(cls, current: Any, stored: Any) -> bool:
        """`True` if a recorded value is the same as the current value.

        Two values are the same only if their canonical forms
        (`FetchRecord.canonical`) are equal and they are equal in value. The
        canonical form shows the type, the time zone, the fields of a
        dataclass, the items of a set and the keys of a mapping. The value
        comparison covers what a shortened `repr` hides: pandas objects
        compare exactly with names and dtypes, numpy arrays by dtype, shape
        and items, mappings, lists, tuples and dataclasses per item, and
        other values with `==`. Two missing values (NaN, `NaT`, `pd.NA`,
        `None`) of the same type are the same.
        """
        try:
            if FetchRecord.canonical(current) != FetchRecord.canonical(stored):
                return False
            return cls._equal(current, stored)
        except Exception:
            return False

    @classmethod
    def _equal(cls, current: Any, stored: Any) -> bool:
        if type(current) is not type(stored):
            return False
        if isinstance(current, Mapping):
            return current.keys() == stored.keys() and all(cls._equal(current[k], stored[k]) for k in current)
        if isinstance(current, (list, tuple)):
            return len(current) == len(stored) and all(cls._equal(c, s) for c, s in zip(current, stored))
        if dataclasses.is_dataclass(current) and not isinstance(current, type):
            return all(
                cls._equal(getattr(current, f.name), getattr(stored, f.name)) for f in dataclasses.fields(current)
            )
        if isinstance(current, (pd.DataFrame, pd.Series, pd.Index)):
            return cls._pandas_equal(current, stored)
        if isinstance(current, np.ndarray):
            return (
                current.dtype == stored.dtype
                and current.shape == stored.shape
                and bool(np.array_equal(current, stored, equal_nan=current.dtype.kind in 'fc'))
            )
        if cls._is_missing(current) or cls._is_missing(stored):
            return cls._is_missing(current) and cls._is_missing(stored)
        return bool(current == stored)

    @staticmethod
    def _pandas_equal(current: Any, stored: Any) -> bool:
        assert_equal = {
            pd.DataFrame: pd.testing.assert_frame_equal,
            pd.Series: pd.testing.assert_series_equal,
        }.get(type(current), pd.testing.assert_index_equal)
        try:
            assert_equal(current, stored, check_exact=True)
            return True
        except AssertionError:
            return False

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """`True` for a scalar missing value: `None`, NaN, `pd.NA` or `NaT`."""
        if not pd.api.types.is_scalar(value):
            return False
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False
