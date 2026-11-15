"""Cache key and the abstract cache interface."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from mesqual.datasets.fetch_context import FetchRecord

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset

Entry = tuple[FetchRecord, pd.Series | pd.DataFrame]
"""One cache entry: `(record, value)`, the fetch record of the fetch that computed the value, and the value."""


@dataclass(frozen=True)
class CacheKey:
    """The key of a cache entry: dataset class, dataset name, flag and parameters.

    The version identity and the config reads are not in the key; the
    entry's record holds them. `Dataset.fetch` uses the entry only if the
    record is still up to date: same version identity and parameters, same
    config reads and files, and each dependency up to date (see
    [Cache](../architecture/other-contracts.md#cache)).

    Attributes:
        dataset_class: `f'{cls.__module__}.{cls.__qualname__}'` of the dataset.
        dataset_name: `name` of the dataset.
        flag: The flag, as string.
        params: The resolved parameters in canonical form
            (`FetchRecord.params_key`). A `param_defaults` entry reaches the
            key only through them.
    """

    dataset_class: str
    dataset_name: str
    flag: str
    params: tuple

    @classmethod
    def of(cls, record: FetchRecord) -> CacheKey:
        """The key of the fetch with the record `record`."""
        return cls(record.dataset_class, record.dataset_name, record.flag, record.params_key(record.params))

    @property
    def digest(self) -> str:
        """Hex digest of the key parts, for example as a file name."""
        return hashlib.sha256(repr(self).encode()).hexdigest()

    def matches(self, dataset: Dataset | None = None, flag: Any = None) -> bool:
        """`True` if the key belongs to `dataset` (class and name) and to `flag`. `None` matches all."""
        if dataset is not None and (self.dataset_class, self.dataset_name) != (dataset._class_path(), dataset.name):
            return False
        return flag is None or self.flag == str(flag)


class Cache(ABC):
    """Abstract storage of fetch results: one entry per cache key.

    An entry is `(record, value)`: the fetch record of the fetch that
    computed the value, and the value. A subclass implements `entry`, `set`,
    `delete` and `list_keys`. The cache does not check entries:
    `Dataset.fetch` checks if the record of the entry is still up to date
    (see [Cache](../architecture/other-contracts.md#cache)).
    """

    @abstractmethod
    def entry(self, key: CacheKey) -> Entry | None:
        """The entry of `key` as `(record, value)`, or `None` if the key has no entry.

        An exception does not fail the fetch: `Dataset.fetch` logs a warning
        and computes the value as on a miss.
        """

    @abstractmethod
    def set(self, key: CacheKey, value: pd.Series | pd.DataFrame, record: FetchRecord) -> None:
        """Store `value` with the fetch record `record` of its fetch as the entry of `key`.

        A key holds one entry, so the new entry replaces the old one. An
        exception does not fail the fetch: `Dataset.fetch` returns the value
        and logs the reason.
        """

    @abstractmethod
    def delete(self, dataset: Dataset | None = None, flag: Any = None) -> None:
        """Delete the entries of the keys of `dataset` (class and name) and `flag`. `None` matches all."""

    @abstractmethod
    def list_keys(self, dataset: Dataset | None = None, flag: Any = None) -> list[CacheKey]:
        """The keys of `dataset` (class and name) and `flag` that have an entry. `None` matches all."""
