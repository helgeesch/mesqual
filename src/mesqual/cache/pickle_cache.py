"""File-based cache with one pickle file per cache key."""

from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from mesqual.cache.cache import Cache, CacheKey, Entry

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset
    from mesqual.datasets.fetch_context import FetchRecord

logger = logging.getLogger(__name__)


class PickleCache(Cache):
    """Cache that stores each cache key in one pickle file in a folder.

    The file name is the digest of the key (`CacheKey.digest`) with the
    suffix `SUFFIX`, so any flag name works, and other files in the folder
    are ignored. The file holds the key and its entry: the fetch record and
    the value. A write goes to a temporary file first and then replaces the
    file.

    A file that cannot be read counts as a key without entry, with a
    warning, and the next `set` of that key replaces it. A `set` that cannot
    write, for example for a value that cannot be pickled, raises and leaves
    no temporary file; `Dataset.fetch` returns the value and logs the
    reason.

    The file stores the path of each file read (`FileRead.path`) relative to
    the folder, and `entry` joins it back onto the current folder. So the
    entries stay up to date when the data and the cache folder move together.
    After a move that changes the relative location of a file, the entry is
    stale and is computed again. A path that holds `..` or is on another
    drive stays absolute.

    Args:
        folder_path: The folder of the files. It is created if it does not exist.

    Examples:
        >>> from mesqual.testing import DummyDataset, DummyScenario
        >>> cache = PickleCache('cache')
        >>> cached = DummyDataset(DummyScenario(), cache=cache)
        >>> prices = cached.fetch('Node.Results.price')
        >>> cache.list_keys(cached, 'Node.Results.price')
        [CacheKey(dataset_class='mesqual.testing.dataset.DummyDataset', dataset_name='base', flag='Node.Results.price', ...)]
        >>> cache.delete(cached)
    """

    SUFFIX = '.entry.pickle'
    """The suffix of the files of the cache. `list_keys` and `delete` read only files with this suffix."""

    def __init__(self, folder_path: str | os.PathLike):
        self._folder_path = Path(folder_path)
        self._folder_path.mkdir(parents=True, exist_ok=True)
        self._absolute_folder = self._folder_path.absolute()

    @property
    def folder_path(self) -> Path:
        """The folder of the files."""
        return self._folder_path

    def entry(self, key: CacheKey) -> Entry | None:
        """The entry of `key`, read from the file of the key. `None` if the file does not exist or cannot be read."""
        content = self._read(self._path(key))
        if content is None:
            return None
        record, value = content['entry']
        return self._with_paths(record, self._current_path), value

    def set(self, key: CacheKey, value: pd.Series | pd.DataFrame, record: FetchRecord) -> None:
        """Replace the file of `key` with the new entry. If the entry cannot be written, raise and keep the file."""
        self._write(key, (self._with_paths(record, self._stored_path), value))

    def delete(self, dataset: Dataset | None = None, flag: Any = None) -> None:
        """Delete the files of the keys of `dataset` and `flag`."""
        for key in self.list_keys(dataset, flag):
            self._path(key).unlink(missing_ok=True)

    def list_keys(self, dataset: Dataset | None = None, flag: Any = None) -> list[CacheKey]:
        """The keys of `dataset` and `flag`, read from the files with `SUFFIX`. Files that cannot be read are skipped."""
        return [key for key in self._keys() if key.matches(dataset, flag)]

    def _path(self, key: CacheKey) -> Path:
        return self._absolute_folder / f'{key.digest}{self.SUFFIX}'

    def _stored_path(self, path: str) -> str:
        """`path` relative to the folder, with `/`. Absolute if it holds `..`, which can follow a link, or has no relative form."""
        if os.path.normpath(path) != path:
            return path
        try:
            return Path(os.path.relpath(path, self._absolute_folder)).as_posix()
        except ValueError:
            return path

    def _current_path(self, path: str) -> str:
        """The absolute path of a stored path: a relative path joined onto the current folder."""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self._absolute_folder, path))

    @classmethod
    def _with_paths(cls, record: FetchRecord, convert: Callable[[str], str]) -> FetchRecord:
        """`record` and its dependencies with `convert` applied to the path of each file read."""
        return dataclasses.replace(
            record,
            file_reads=tuple(dataclasses.replace(read, path=convert(read.path)) for read in record.file_reads),
            dependencies=tuple(cls._with_paths(dependency, convert) for dependency in record.dependencies),
        )

    @staticmethod
    def _read(path: Path) -> dict | None:
        """The content of the file at `path`, or `None` if it does not exist or cannot be read."""
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as file:
                content = pickle.load(file)
            record, value = content['entry']
            return {'key': content['key'], 'entry': (record, value)}
        except Exception as error:
            logger.warning(f'Cache file {path} cannot be read and is ignored: {type(error).__name__}: {error}')
            return None

    def _write(self, key: CacheKey, entry: Entry) -> None:
        handle, temporary = tempfile.mkstemp(dir=self._absolute_folder, suffix='.tmp')
        try:
            with os.fdopen(handle, 'wb') as file:
                pickle.dump({'key': key, 'entry': entry}, file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, self._path(key))
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _keys(self) -> list[CacheKey]:
        contents = (self._read(path) for path in sorted(self._absolute_folder.glob(f'*{self.SUFFIX}')))
        return [content['key'] for content in contents if content is not None]
