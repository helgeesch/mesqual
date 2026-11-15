"""Fetch context: what each fetch used, recorded at runtime.

- `FetchContext` holds the stack of open fetches. `_fetch` calls
  `FetchContext.record_file_read` for each source file.
- `FetchRecord` holds what one fetch used: the call, the resolved
  parameters, the version identity, the config reads, the file reads
  (`FileRead`) and the records of the nested fetches.

`mesqual.datasets.dependency_check.DependencyCheck` decides if a stored
record is still up to date.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import dataclasses
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Mapping

if TYPE_CHECKING:
    from mesqual.datasets.dataset_config import DatasetConfig


class _Inherit:
    """Type of the `INHERIT` marker. There is one instance."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self):
        return _Inherit, ()

    def __repr__(self) -> str:
        return 'INHERIT'


INHERIT = _Inherit()
"""`FetchRecord.call_config` of a call that got the effective config of the enclosing fetch as its config."""


class Unset:
    """Type of the `UNSET` marker, for the type hints of typed `fetch` overrides. There is one instance."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self):
        return Unset, ()

    def __repr__(self) -> str:
        return '<from param_defaults or default>'


UNSET = Unset()
"""Default of the typed dataset parameters of a `fetch` override: the call does not give the parameter.

The parameter then resolves from `param_defaults` of the effective config,
else from the default of its dimension (`Dataset.fetch_params`).
"""


def snapshot(value: Any) -> Any:
    """Return a deep copy of `value` for a record, so a later in-place change does not reach the record.

    A value that cannot be deep-copied is kept as is.
    """
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


@dataclass(frozen=True)
class FileRead:
    """The state of a file: path, size, modification time and, for a small file, a content hash.

    The cache records one `FileRead` for each file that a fetch read. A cache entry is up to
    date only if each of these files is unchanged (`is_unchanged`).

    Attributes:
        path: Absolute path of the file: the path as given, made absolute
            with the working directory at record time. Symbolic links and
            `..` are kept, so a link that points to another file is changed.
        size: Size in bytes.
        mtime_ns: Modification time in nanoseconds.
        sha256: Hex digest of the content, for a file of at most
            `HASH_MAX_BYTES` bytes. `None` for a larger file, or if read
            with `with_hash=False`.
    """

    HASH_MAX_BYTES: ClassVar[int] = 16 * 1024**2
    """Files up to this size in bytes get a content hash. A larger file is checked by size and modification time."""

    path: str
    size: int
    mtime_ns: int
    sha256: str | None = None

    @classmethod
    def of(cls, path: str | os.PathLike, with_hash: bool = True) -> FileRead:
        """Return the current state of the file at `path`.

        Args:
            path: The path of the file.
            with_hash: `False` gives no content hash, also for a small file.

        Raises:
            ValueError: `path` is a directory.
            OSError: The file does not exist or cannot be read.
        """
        absolute = Path(path).absolute()
        if absolute.is_dir():
            raise ValueError(f'{absolute} is a directory. Give the files in it one by one.')
        stat = absolute.stat()
        digest = cls._sha256(absolute) if with_hash and stat.st_size <= cls.HASH_MAX_BYTES else None
        return cls(str(absolute), stat.st_size, stat.st_mtime_ns, digest)

    def is_unchanged(self) -> bool:
        """`True` if the file still exists with the same size and content.

        A file with a recorded content hash is unchanged if the size and
        the hash are the same; the modification time does not count. A file
        without content hash is unchanged if the size and the modification
        time are the same.
        """
        try:
            stat = os.stat(self.path)
            if stat.st_size != self.size:
                return False
            if self.sha256 is None:
                return stat.st_mtime_ns == self.mtime_ns
            return self._sha256(Path(self.path)) == self.sha256
        except OSError:
            return False

    @property
    def key(self) -> str | tuple[str, int, int]:
        """Identity of the file state without the directory.

        A file with a content hash (at most `HASH_MAX_BYTES`, read with
        `with_hash=True`) has the hash as key, without the file name. Any
        other file has the key `(file name, size, mtime_ns)`, so two copies of
        it have the same key only if they have the same modification time.
        A fingerprint pairs each key with the name of the file relative to
        the source folder, so a rename or a swap of two files changes it.
        """
        return self.sha256 or (Path(self.path).name, self.size, self.mtime_ns)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, 'rb') as file:
            for chunk in iter(lambda: file.read(1024**2), b''):
                digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class FetchRecord:
    """What one `fetch` call used: the call as given, the resolved parameters, the config reads and the dependencies.

    A record has two views of a call:

    - The call as given: `call_kwargs` and `call_config`. They identify the call.
    - The parameters after resolution: `params`. `_fetch` got these values.

    Every keyword argument of `fetch` is a parameter. A flag spec declares
    flag parameters (`FlagSpec.params`). A dataset class declares dataset
    parameters (`Dataset.fetch_params`). Both resolve the same way: the
    keyword argument, then `param_defaults` of the effective config, then the
    `default` of the dimension.

    Examples:
        The spec of `'Node.Results.price'` declares the flag parameter
        `solution` with the default `'final'`. The instance config sets
        `param_defaults={'solution': 'first'}`.

        >>> high = study.scen.get_dataset('high_demand')
        >>> high.set_instance_config_kwargs(param_defaults={'solution': 'first'})
        >>> prices = high.fetch('Node.Results.price', config={'use_cache': False})
        >>> record = high.get_fetch_record('Node.Results.price', config={'use_cache': False})
        >>> record.call_kwargs, record.call_config
        ({}, {'use_cache': False})
        >>> record.params, record.param_default_reads
        ({'solution': 'first'}, {'solution': 'first'})

        `solution` is not in the call, so it comes from `param_defaults`.
        Without that entry, it is `'final'` and `param_default_reads` is empty.

    The records form a tree. The record of a fetch lists the fetches that it
    calls directly as its dependencies, and each of them lists its own. Every
    nested fetch has its own record, also a call that a platform dataset
    routes to an interpreter, a link collection to a child, or an override to
    its base. `iter_dependencies` walks the tree.

    Attributes:
        dataset_class: `f'{cls.__module__}.{cls.__qualname__}'` of the dataset.
        dataset_name: `name` of the dataset.
        flag: The fetched flag, as string.
        call_kwargs: The keyword arguments of the call, as given. Defaults
            are not filled in.
        call_config: The config override of the call: its explicit fields
            with their values. `DatasetConfig.CLEAR` stays as is. `INHERIT`
            if the call got the effective config of the enclosing fetch, as a
            collection passes it to its children. The call then follows the
            current config of its parent record.
        params: All parameters after resolution: the flag parameters and the
            dataset parameters, with defaults filled in. For a collection:
            its own dataset parameters (`fetch_params`), resolved, and the
            flag parameters and the dataset parameters of its children that
            the call gave. A collection resolves neither of the last two.
        version: Version identity of the flag (`Dataset.version_identity`).
        param_default_reads: The entries of `param_defaults` that parameter
            resolution used: one per parameter that the call did not give
            and that `param_defaults` names.
        config_reads: Each config field that `_fetch` or the mesqual
            post-processing read from `effective_config`, with the value read.
            Reading the whole config (`dataclasses.asdict`, `vars`, `copy`,
            `pickle`, `repr`) records all fields. `param_defaults` is here
            only if `_fetch` read the field itself.
        dependencies: The records of the fetches that this fetch called
            directly, in the order of the calls, each call key once.
        raised: `True` if the call raised.
        file_reads: The state of each file that `_fetch` recorded with
            `FetchContext.record_file_read`, one `FileRead` per path.
        reads_given_params: `True` if the fetch passed its given parameters
            (`given_params`) on to its children, as a collection does.

    The values in `call_kwargs`, `call_config`, `params`,
    `param_default_reads` and `config_reads` are deep copies, taken when the
    call recorded them. A record hashes by its `key`, so records work in
    sets. `DependencyCheck` (`mesqual.datasets.dependency_check`) decides if a
    stored record is up to date.
    """

    dataset_class: str
    dataset_name: str
    flag: str
    call_kwargs: Mapping[str, Any]
    call_config: Mapping[str, Any]
    params: Mapping[str, Any]
    version: tuple
    param_default_reads: Mapping[str, Any] = field(default_factory=dict)
    config_reads: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[FetchRecord, ...] = ()
    raised: bool = False
    file_reads: tuple[FileRead, ...] = ()
    reads_given_params: bool = False

    def __hash__(self) -> int:
        return hash(self.key)

    @property
    def key(self) -> tuple:
        """Identity of the call: dataset class, dataset name, flag, resolved parameters, call kwargs and call config.

        Two fetches of one flag with different keyword arguments or config
        overrides are two keys.
        """
        call_config = repr(INHERIT) if self.call_config is INHERIT else self.params_key(self.call_config)
        return (
            self.dataset_class,
            self.dataset_name,
            self.flag,
            self.params_key(self.params),
            self.params_key(self.call_kwargs),
            call_config,
        )

    @property
    def given_params(self) -> dict[str, Any]:
        """The resolved parameters that the call or `param_defaults` gave: names in `call_kwargs` or `param_default_reads`.

        A parameter that resolved to the default of its dimension is not here.
        """
        return {
            name: value
            for name, value in self.params.items()
            if name in self.call_kwargs or name in self.param_default_reads
        }

    def iter_dependencies(self) -> Iterator[FetchRecord]:
        """Walk the dependency tree depth first, parents before children.

        A record that two parents call appears once under each parent.
        """
        for dependency in self.dependencies:
            yield dependency
            yield from dependency.iter_dependencies()

    @classmethod
    def params_key(cls, params: Mapping[str, Any]) -> tuple:
        """Canonical form of parameters: pairs of name and canonical text of the value, sorted by name.

        The text is the `repr` of the value, with the items of a dict and of
        a set sorted, so the insertion order does not change it.
        """
        return tuple(sorted((name, cls.canonical(value)) for name, value in params.items()))

    @classmethod
    def canonical(cls, value: Any) -> str:
        """Canonical text of a value: its `repr`, with the items of dicts and sets sorted, also nested."""
        if isinstance(value, Mapping):
            items = sorted(f'{cls.canonical(k)}: {cls.canonical(v)}' for k, v in value.items())
            return f'{type(value).__name__}({{{", ".join(items)}}})'
        if isinstance(value, (set, frozenset)):
            return f'{type(value).__name__}({{{", ".join(sorted(map(cls.canonical, value)))}}})'
        if isinstance(value, (list, tuple)):
            return f'{type(value).__name__}([{", ".join(map(cls.canonical, value))}])'
        return repr(value)


class FetchFrame:
    """One open `fetch` call in the fetch context. Collects config reads and nested fetch records.

    Args:
        record: The record of the call without config reads and dependencies.
        dataset: The dataset whose `fetch` opened the frame.

    Attributes:
        dataset: The dataset whose `fetch` opened the frame.
        watched_config: The effective config that `_fetch` gets, set by the dataset.
        config_reads: The config fields read so far, with deep copies of the values.
        file_reads: The `FileRead` of each recorded file, by path.
        reads_given_params: `True` once the fetch passed its given
            parameters on to its children (`read_given_params`).
        closed: `True` after the frame closed. A closed frame records nothing.
        stores: `True` if the fetch can store its result in a cache, set
            by the dataset. See `FetchContext.record_file_read`.
        raised: `True` if the call raised.
    """

    def __init__(self, record: FetchRecord, dataset: Any = None):
        self._record = record
        self.dataset = dataset
        self.watched_config: DatasetConfig | None = None
        self.stores = False
        self.config_reads: dict[str, Any] = {}
        self.file_reads: dict[str, FileRead] = {}
        self.reads_given_params = False
        self._dependencies: dict[tuple, FetchRecord] = {}
        self.closed = False
        self.raised = False
        self._final_record: FetchRecord | None = None

    @property
    def record(self) -> FetchRecord:
        """The record of the call, without config reads and dependencies."""
        return self._record

    @property
    def dependencies(self) -> tuple[FetchRecord, ...]:
        """The records of the fetches that this fetch called directly so far."""
        return tuple(self._dependencies.values())

    def add_dependencies(self, records: Iterable[FetchRecord]) -> None:
        """Add records of direct nested fetches, for example the record of a cache hit."""
        for record in records:
            self._dependencies.setdefault(record.key, record)

    def record_config_read(self, name: str, value: Any) -> None:
        """Record that `_fetch` read the config field `name` with `value`. The record keeps a deep copy."""
        if not self.closed and FetchContext.is_recording():
            self.config_reads[name] = snapshot(value)

    def read_given_params(self) -> dict[str, Any]:
        """Return `FetchRecord.given_params` of this call and mark the record with `reads_given_params`."""
        self.reads_given_params = True
        return self._record.given_params

    def add_stored(self, stored: FetchRecord) -> None:
        """Take over the reads and dependencies of the stored record of a cache hit."""
        self.config_reads.update(stored.config_reads)
        self.file_reads.update({file_read.path: file_read for file_read in stored.file_reads})
        self.reads_given_params = self.reads_given_params or stored.reads_given_params
        self.add_dependencies(stored.dependencies)

    def to_record(self) -> FetchRecord:
        """Return the record of this fetch with its reads and dependencies. After close, always the same object."""
        if self._final_record is not None:
            return self._final_record
        return dataclasses.replace(
            self._record,
            config_reads=dict(self.config_reads),
            dependencies=self.dependencies,
            raised=self.raised,
            file_reads=tuple(self.file_reads.values()),
            reads_given_params=self.reads_given_params,
        )

    def _close_into(self, parent: FetchFrame | None) -> None:
        self._final_record = self.to_record()
        self.closed = True
        if parent is not None:
            parent.add_dependencies([self._final_record])


class FetchContext:
    """The stack of open fetch frames of the current context.

    In `_fetch`, call `FetchContext.record_file_read(path)` for each source
    file, so an edited file makes the cache entries that read it stale. The
    other methods are for mesqual itself.

    The stack lives in context variables. Code in a thread started inside a fetch
    runs with `contextvars.copy_context().run(...)`, else its fetches count as
    outermost and are not recorded in the open frames.
    """

    _frames: contextvars.ContextVar[tuple[FetchFrame, ...]] = contextvars.ContextVar('mesqual_fetch_frames', default=())
    _recording: contextvars.ContextVar[bool] = contextvars.ContextVar('mesqual_fetch_recording', default=True)

    @classmethod
    def current(cls) -> FetchFrame | None:
        """The innermost open frame, or `None` outside of a fetch."""
        frames = cls._frames.get()
        return frames[-1] if frames else None

    @classmethod
    def is_outermost(cls) -> bool:
        """`True` if no fetch is open, so the next fetch is the outermost one."""
        return not cls._frames.get()

    @classmethod
    def is_recording(cls) -> bool:
        """`False` while mesqual prepares a fetch, so its own config reads are not recorded."""
        return cls._recording.get()

    @classmethod
    @contextlib.contextmanager
    def frame(cls, record: FetchRecord, dataset: Any = None) -> Iterator[FetchFrame]:
        """Open a frame for one fetch call of `dataset`. On exit, its record goes to the enclosing frame.

        If the call raises, the record has `raised=True`.
        """
        frame = FetchFrame(record, dataset)
        parent = cls.current()
        frames_token = cls._frames.set(cls._frames.get() + (frame,))
        recording_token = cls._recording.set(True)
        try:
            yield frame
        except BaseException:
            frame.raised = True
            raise
        finally:
            cls._recording.reset(recording_token)
            cls._frames.reset(frames_token)
            frame._close_into(parent)

    @classmethod
    @contextlib.contextmanager
    def internal(cls) -> Iterator[None]:
        """Suspend the recording of config reads, for mesqual code that prepares a fetch."""
        token = cls._recording.set(False)
        try:
            yield
        finally:
            cls._recording.reset(token)

    @classmethod
    def record_file_read(cls, path: str | os.PathLike) -> None:
        """Record that the open fetch read the file at `path`: path, size, modification time, content hash.

        Call it in `_fetch` for each source file that the result depends on,
        before the file is read, so a change during the read makes the
        entry stale.
        Call it in the thread of the fetch, or in a thread that runs with
        `contextvars.copy_context().run(...)`.
        A cache entry is up to date only if each recorded file is unchanged,
        so an edited file makes stale exactly the entries whose fetches read
        it. Outside of a fetch, it does nothing. The content hash is computed
        only if a fetch on the stack can store its result in a cache.

        Raises:
            ValueError: `path` is a directory.
            OSError: The file does not exist or cannot be read.

        Examples:
            ```python
            def _fetch(self, flag, effective_config, **params):
                FetchContext.record_file_read(self.path)
                return pd.read_csv(self.path)
            ```
        """
        frame = cls.current()
        if frame is not None and not frame.closed and cls.is_recording():
            file_read = FileRead.of(path, with_hash=any(f.stores for f in cls._frames.get()))
            frame.file_reads[file_read.path] = file_read

    @staticmethod
    def watch(config: DatasetConfig, frame: FetchFrame) -> DatasetConfig:
        """Return a copy of `config` whose field reads go to `frame`, and store it as `frame.watched_config`."""
        frame.watched_config = config._watched(frame.record_config_read)
        return frame.watched_config
