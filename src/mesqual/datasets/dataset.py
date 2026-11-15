from __future__ import annotations

import copy
import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterable, Type, Union

import pandas as pd

from mesqual.typevars import DatasetConfigType, FlagType, FlagIndexType
from mesqual.cache import Cache, CacheKey, Entry
from mesqual.datasets.fetch_context import (
    INHERIT,
    UNSET,
    FetchContext,
    FetchFrame,
    FetchRecord,
    Unset,
    snapshot,
)
from mesqual.datasets.dependency_check import DependencyCheck
from mesqual.datasets.describe import Availability, DatasetTreeDescriber, FlagDescriber, FlagInfo
from mesqual.datasets.time_index_filter import TimeIndexFilter
from mesqual.enums import ItemTypeEnum
from mesqual.flag.dimension import Dimension
from mesqual.flag.exceptions import FlagUnavailableError
from mesqual.flag.flag_index import FlagIndex
from mesqual.flag.flag_spec import FlagSpec
from mesqual.flag.layers import LayerSet
from mesqual.flag.schema import FrameSchema
from mesqual.utils.pandas_utils.move_level import move_top_level_last

if TYPE_CHECKING:
    from mesqual.datasets.dataset_collection import DatasetLinkCollection
    from mesqual.kpis.kpi import KPI
    from mesqual.kpis.definitions.base import KPIDefinition
    from mesqual.kpis.collection import KPICollection

logger = logging.getLogger(__name__)


class _UnknownConfigFieldWarnings:
    """Process-wide record of the unknown config fields that were already logged.

    One warning per config class and field name, for all dataset instances.
    """

    _warned: set[tuple[type, str]] = set()

    @classmethod
    def warn(cls, dataset_name: str, config: DatasetConfigType, declared_by_parents: frozenset[str]) -> None:
        """Log a warning for the unknown fields of `config` not yet logged for its class.

        Args:
            dataset_name: Name of the dataset that received `config`.
            config: Effective config of a fetch.
            declared_by_parents: Field names that the config classes of the
                parent datasets declare. They never warn.
        """
        config_type = type(config)
        unknown = {
            name for name in config._unknown_fields() - declared_by_parents if (config_type, name) not in cls._warned
        }
        if not unknown:
            return
        logger.warning(
            f'Dataset {dataset_name}: config fields {sorted(unknown)} are not defined by '
            f'{config_type.__name__} and have no effect.'
        )
        cls._warned |= {(config_type, name) for name in unknown}

    @classmethod
    def clear(cls) -> None:
        """Forget all logged fields, so they warn again."""
        cls._warned.clear()


class _DuplicateIndexWarnings:
    """Process-wide record of the duplicate index warnings that were already logged.

    One warning per dataset class and flag, for all dataset instances.
    """

    _warned: set[tuple[type, str]] = set()

    @classmethod
    def warn(cls, dataset: Dataset, flag: FlagType, dropped_rows: int, keep: str) -> None:
        """Log a warning about dropped duplicate rows, if not yet logged for this dataset class and flag.

        Args:
            dataset: Dataset whose result has duplicate index entries.
            flag: Fetched flag.
            dropped_rows: Number of rows dropped.
            keep: The `keep_duplicate_indices` value that applied.
        """
        key = (type(dataset), str(flag))
        if key in cls._warned:
            return
        cls._warned.add(key)
        logger.warning(
            f'Dataset {dataset.name}: flag {flag} has duplicate index entries. '
            f'Dropped {dropped_rows} rows and kept the {keep} row of each duplicate. '
            f"Set keep_duplicate_indices='all' to keep all rows."
        )

    @classmethod
    def clear(cls) -> None:
        """Forget all logged warnings, so they warn again."""
        cls._warned.clear()


class _NotStoredLogs:
    """Process-wide record of the flags whose reason for not being cached was already logged.

    One INFO log per dataset class and flag, for all dataset instances.
    """

    _logged: set[tuple[type, str]] = set()

    @classmethod
    def log(cls, dataset: Dataset, flag: FlagType, reason: str) -> None:
        """Log at INFO why the result of `flag` is not cached, if not yet logged for this dataset class and flag."""
        key = (type(dataset), str(flag))
        if key in cls._logged:
            return
        cls._logged.add(key)
        logger.info(f'{dataset.name} {flag} is not cached: {reason}.')

    @classmethod
    def clear(cls) -> None:
        """Forget all logged flags, so they log again."""
        cls._logged.clear()


class Dataset(Generic[DatasetConfigType, FlagType, FlagIndexType], ABC):
    """Base class of all datasets: one `fetch(flag)` interface for all data.

    A scenario, a scenario merged from several runs or sources, a collection of
    scenarios and a scenario comparison are all datasets.

    A dataset declares its flags with flag specs in `flag_specs` and
    implements `_fetch(flag, effective_config, **params)`. `accepted_flags`
    comes from the specs. A dataset can have attributes (scenario metadata), a
    KPI collection and a cache of fetch results.

    Attributes:
        name (str): Name of the dataset.
        kpi_collection (KPICollection): The KPIs of the dataset.

    Examples:
        A dataset class:

        ```python
        class NodeResults(Dataset):
            flag_specs = (NODE_MODEL_SPEC, NODE_PRICE_SPEC)

            def _fetch(self, flag, effective_config, **params):
                ...
        ```

        A fetch:

        >>> prices = dataset.fetch('Node.Results.price')
    """

    flag_specs: ClassVar[tuple[FlagSpec, ...]] = ()
    """Flag specs of this dataset. `accepted_flags` and the flag index come from them."""

    description: str | None = None
    """Optional free text about the dataset. A platform class or a study sets it, on the class or the instance.
    `describe_datasets()` shows it."""

    _AFTER_CACHE_FIELDS: ClassVar[tuple[str, ...]] = ('time_index_filter', 'granularity')
    """Config fields that the outermost fetch applies after the cache. `_fetch` gets them as `None`."""

    fetch_params: ClassVar[tuple[Dimension, ...]] = ()
    """Dataset parameters: keyword arguments of `fetch` that the dataset class uses itself.

    A class has the dataset parameters of its base classes and adds its own:
    `fetch_params` of a subclass lists only the new ones, and the class
    gets the merged tuple.

    A flag spec declares the parameters of one flag (`FlagSpec.params`).
    `fetch_params` declares the parameters of every fetch of this dataset
    class, for example `comparison_type` of `DatasetComparison`. Both kinds
    resolve the same way: the keyword argument, then `param_defaults` of the
    effective config, then the `default` of the dimension. The resolved value
    goes to `_fetch` as a keyword argument.

    A name may not be both a flag parameter and a dataset parameter of one
    dataset. A collection also accepts the dataset parameters of its
    children. It passes each child the flag parameters of the call and the
    dataset parameters that the child accepts and that the call or the
    `param_defaults` of the collection gave, its own ones included (for
    example `comparison_type` to the comparisons of a comparison of
    comparisons). A value in the call and the same value in `param_defaults`
    reach the child alike. The collection never passes the default of a
    dimension.

    Examples:
        ```python
        class NodePrices(Dataset):
            flag_specs = (NODE_PRICE_SPEC,)
            fetch_params = (Dimension('currency', 'Currency of the prices.', {
                'EUR': 'euro', 'USD': 'US dollar',
            }, default='EUR'),)

            def _fetch(self, flag, effective_config, currency, **params):
                ...

        NodePrices().fetch('Node.Results.price', currency='USD')
        ```

    A subclass may override `fetch` to show its dataset parameters in the
    IDE: one typed keyword-only argument per dataset parameter, with the
    default `UNSET`. The override passes on to `super().fetch` only the
    arguments that are not `UNSET` (`_given`). So a parameter that the call
    leaves out still resolves from `param_defaults`, and `call_kwargs` holds
    only what the caller gave.
    """

    _fetch_param_names_of_all_classes: ClassVar[set[str]] = set()
    """Names of the dataset parameters that any `Dataset` subclass declares.

    A `param_defaults` entry of a class config (`set_class_config`) with one
    of these names never raises as unknown, so a global config can set
    `comparison_type` for all datasets.
    """

    _reads_source: ClassVar[bool] = True
    """`True`: `fetch` raises `ValueError` up front for an unavailable flag. Datasets that read the data
    and platform datasets (with the answer of the interpreter chain). `False` for other collections,
    which fetch only from children on which the flag is not unavailable."""

    _resolves_flag_params: ClassVar[bool] = True
    """`False` for collections: they check flag parameter names and pass the
    explicit flag parameters to their children, which resolve them with their
    own config. A collection always resolves its own `fetch_params`."""

    _warns_about_unknown_config_fields: bool = True
    """Log a warning for config keys that the config type does not define.

    The warning is logged once per config class and key in the process, not
    once per dataset instance. So one undeclared key, fetched across many
    scenarios and interpreters, gives one warning. A key that the config
    class of the dataset or of a parent dataset declares never warns, so a
    key of a platform config does not warn on any interpreter. `False` for
    collections, which pass such keys on to their children."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'fetch_params' in cls.__dict__:
            merged = {d.name: d for base in reversed(cls.__bases__) for d in getattr(base, 'fetch_params', ())}
            merged.update({d.name: d for d in cls.__dict__['fetch_params']})
            cls.fetch_params = tuple(merged.values())
        Dataset._fetch_param_names_of_all_classes.update(d.name for d in cls.fetch_params)

    def __init__(
        self,
        name: str = None,
        parent_dataset: Dataset = None,
        attributes: dict = None,
        cache: Cache = None,
        config: DatasetConfigType = None,
    ):
        """
        Initialize a new Dataset instance.

        Args:
            name: Human-readable identifier. If None, auto-generates from class name
            parent_dataset: Optional parent dataset for hierarchical relationships
            attributes: Dictionary of metadata attributes for the dataset. Keys must be
                of type str, values of type bool, int, float or str.
            cache: Optional cache of fetch results
            config: Configuration object controlling dataset behavior

        Raises:
            TypeError: If an attribute key or value has an invalid type.
            ValueError: A time series spec names a model flag without model spec.
            AmbiguousOverrideError: Two specs give the same flag name.
            ValueError: A name is a flag parameter and a dataset parameter
                (`fetch_params`).
        """
        self.name = name or f'{self.__class__.__name__}_{str(id(self))}'
        self._flag_index: FlagIndex | None = None
        self._parent_dataset = parent_dataset
        self._attributes: dict = dict()
        self._add_attributes(attributes or dict())
        self._cache = cache
        self._config = config
        self._fetch_records: dict[tuple, FetchRecord] = {}
        self._fetched_flags: set[str] = set()
        self._flag_index = self._build_flag_index()
        self._check_param_name_clash()

        from mesqual.kpis.collection import KPICollection

        self.kpi_collection: KPICollection = KPICollection()

    def fetch(
        self,
        flag: FlagType,
        config: dict | DatasetConfigType = None,
        *,
        granularity: str | pd.Timedelta | None | Unset = UNSET,
        **kwargs: Any,
    ) -> pd.Series | pd.DataFrame:
        """Fetch the data of one flag.

        `fetch` takes two kinds of input:

        - `config`: overrides of config fields for this call.
        - `**kwargs`: parameters. Every keyword argument is a parameter. A
          flag spec declares flag parameters (`FlagSpec.params`), for example
          `solution`. The dataset class declares dataset parameters
          (`fetch_params`), for example `comparison_type` of a comparison.

        Each parameter resolves the same way, first match wins:

        1. The keyword argument of this call.
        2. `param_defaults` of the effective config.
        3. The `default` of the dimension.

        `_fetch` gets the resolved parameters as keyword arguments. A
        collection resolves only its own dataset parameters. It passes the
        flag parameters of the call to its children. It gives each child
        the dataset parameters that the child accepts and that the call or
        the `param_defaults` of the collection gave, its own ones included,
        never the default of a dimension (`_kwargs_for_child`).

        Config resolution, later wins: config class defaults, the class
        config of each class in the MRO (base first), the instance config,
        and the `config` of this call. Only the explicit fields of a layer
        override lower layers (the keys of a dict, or the explicit fields of
        a config object), also with a `None` value. `DatasetConfig.CLEAR`
        resets a field to its default.

        Steps:

        1. Raise `UnknownFlagError` if no flag spec declares the flag,
           and `ValueError` if this dataset does not accept it or if
           a dataset that reads the data itself (not a collection)
           has the flag unavailable (`_is_available` is `False`).
           Collections fetch only from children on which the flag is
           not unavailable.
        2. Resolve the effective config.
        3. Resolve the parameters.
        4. Open a frame in the fetch context. Nested fetches and the
           config fields that `_fetch` reads are recorded in it
           (`get_fetch_record`).
        5. Call `_fetch` with the resolved parameters, or read the
           cache, and post-process.
        6. In the outermost call, apply the time index filter to a time
           series flag, then convert it to the `granularity` of the
           effective config.

        Settings applied after the cache (`time_index_filter`,
        `granularity`): they apply only in the outermost `fetch` call, from
        its effective config, and only to time series flags, first the
        filter, then the conversion. `_fetch` gets them as `None`, so a
        cached value is never filtered or converted. Nested fetches
        (collection children, interpreters, layer flags) return unfiltered
        native data: such a setting in the config of a child dataset has no
        effect when the child is fetched through a collection. The fetch
        depth lives in a context variable: run code in a thread started
        inside a fetch with `contextvars.copy_context().run(...)`, else its
        fetches count as outermost.

        The config field `granularity` converts a time series flag to
        another time resolution with `TimeSeriesGranularityConverter` and
        the quantity type of the flag: intensive values are averaged or
        repeated, extensive values summed or split. `None` means the native
        resolution. On a row `MultiIndex` it converts the datetime level per
        combination of the other levels. Model and other flags are returned
        unchanged. The argument `granularity` is a short form of
        `config={'granularity': ...}` in the outermost call. In a nested
        fetch it has no effect, so the call keeps its config as given.

        Args:
            flag: The flag. It must be in `accepted_flags` or match a flag
                family of the dataset.
            config: Config overrides for this call: a dict with config field
                names as keys, or a config object of the config type of the
                dataset. A platform config may add its own fields.
            granularity: Target time resolution of a time series flag, a
                pandas frequency such as `'15min'` or `'1h'`. `None`: the
                native resolution. Left out: the `granularity` of the
                effective config. It overrides the `config` of the call.
                No effect in a nested fetch.
            **kwargs: Parameters: flag parameters of the spec and dataset
                parameters of the dataset class (for a collection, also of
                its children).

        Returns:
            DataFrame or Series with the data of the flag.

        Raises:
            UnknownFlagError: No flag spec declares the flag.
            FlagUnavailableError: The flag is not available
                (`is_flag_available` is `False`). A subclass of `ValueError`.
            ValueError: The dataset does not accept the flag; a keyword
                argument is no declared parameter; a parameter is missing or
                has a value that its dimension does not allow;
                `param_defaults` names an unknown parameter (outermost call
                only). A name in the instance config or in `config` must be
                a flag parameter of the flag index or a dataset parameter of
                this dataset (for a collection, also of its children). A
                name in a class config (`set_class_config`) may also be a
                dataset parameter of any `Dataset` subclass. The
                `granularity` of the effective config is not a frequency
                such as `'15min'`.
            TypeError: A `granularity` conversion of a result whose row
                index has no datetime level.

        Examples:
            A time series in another resolution. The base scenario has hourly
            rows, then quarter-hourly rows. The result has one row per hour:

            >>> hourly = dataset.fetch('Node.Results.price', granularity='1h')
            >>> len(dataset.fetch('Node.Results.price')), len(hourly)
            (332, 119)
        """
        is_outermost = FetchContext.is_outermost()
        enclosing = FetchContext.current()
        with FetchContext.internal():
            if granularity is not UNSET and is_outermost:
                config = self._with_granularity(config, granularity)
            spec = self._spec_of_accepted_flag(flag)
            if self._reads_source:
                self._raise_if_unavailable(flag)
            if is_outermost:
                self.flag_index.check_references()
            if is_outermost:
                self._check_param_default_names(config)
            record, effective_config, params = self._start_call(
                spec,
                flag,
                config,
                self._call_config(config, enclosing),
                kwargs,
            )
            if is_outermost:
                self._check_granularity(effective_config.granularity)
            self._warn_about_unknown_config_fields(effective_config)
        shapes_output = is_outermost and spec.item_type is ItemTypeEnum.TimeSeries
        target_granularity = effective_config.granularity if shapes_output else None
        try:
            with FetchContext.frame(record, self) as frame:
                self._check_dataset_values(spec, flag, params)
                data = self._fetch_from_cache_or_source(flag, effective_config, frame, **params)
            self._fetched_flags.add(str(flag))
        finally:
            self._fetch_records[record.key] = frame.to_record()
        if shapes_output:
            data = self._apply_time_index_filter(data, effective_config)
            data = self._convert_granularity(data, flag, target_granularity)
        return data

    def get_fetch_record(
        self,
        flag: FlagType,
        config: dict | DatasetConfigType = None,
        *,
        granularity: str | pd.Timedelta | None | Unset = UNSET,
        **kwargs: Any,
    ) -> FetchRecord | None:
        """Return the record of the last fetch with the same flag, keyword arguments, config and resolved parameters.

        A fetch that raised also leaves a record, with `raised=True`.

        The record lists the fetches that the call made directly, each with
        its own record, and the config fields that `_fetch` read. The lookup
        uses the record key (`FetchRecord.key`): the keyword arguments and
        the call config as given, and the parameters, resolved as in `fetch`.

        Args:
            flag: The flag.
            config: Fetch-time config, as in `fetch`. `INHERIT` finds a call
                that got the effective config of its enclosing fetch; the
                parameters then resolve with the config of this dataset. So
                a child record whose parameters came from the `param_defaults`
                of the parent config is not found here. Reach it through the
                `dependencies` of the parent record.
            granularity: Applies as in an outermost fetch. To look up a
                nested call, leave it out. No effect with `config=INHERIT`.
            **kwargs: Parameters, as in `fetch`.

        Returns:
            The record, or `None` if the flag was not fetched with these arguments.

        Examples:
            >>> high = study.scen.get_dataset('high_demand')
            >>> prices = high.fetch('Node.Results.price', solution='first')
            >>> high.get_fetch_record('Node.Results.price', solution='first').call_kwargs
            {'solution': 'first'}
            >>> high.get_fetch_record('Node.Results.price') is None
            True
        """
        with FetchContext.internal():
            if config is INHERIT:
                layer, call_config = None, INHERIT
            else:
                layer = config if granularity is UNSET else self._with_granularity(config, granularity)
                call_config = self._call_config(layer, None)
            spec = self.flag_index.get_spec(flag)
            record, *_ = self._start_call(spec, flag, layer, call_config, kwargs)
        return self._fetch_records.get(record.key)

    def _start_call(
        self,
        spec: FlagSpec,
        flag: FlagType,
        config: dict | DatasetConfigType | None,
        call_config: Any,
        call_kwargs: dict[str, Any],
    ) -> tuple[FetchRecord, DatasetConfigType, dict[str, Any]]:
        """Resolve the config and the parameters of a call and build its record, before its frame opens.

        `fetch`, `get_fetch_record` and `_rebuild_call` use it, so a stored
        call and its rebuild resolve the same way.

        Args:
            spec: The spec of the flag.
            flag: The flag.
            config: The fetch-time config layer of the call.
            call_config: `FetchRecord.call_config` of the record.
            call_kwargs: The keyword arguments of the call, as given.

        Returns:
            The record, the effective config and the resolved parameters.
        """
        effective_config = self._prepare_config(config)
        params, param_default_reads = self._resolve_params(spec, flag, call_kwargs, effective_config)
        record = self._new_record(flag, call_config, call_kwargs, params, param_default_reads)
        return record, effective_config, params

    def version_identity(self, flag: FlagType) -> tuple:
        """Return the version identity of a flag: one entry per link.

        A dataset gives `((type(self).__qualname__, spec version),)`. A
        platform dataset gives one pair per link of the interpreter chain of
        the flag. The answering interpreter of a platform dataset gives the
        identity of its platform dataset, so a version change of an override
        changes the records of the override and of everything that calls it.
        A collection gives one `(class path, name)` pair per child that it can
        read the flag from; its children record their own identities. A concat collection
        adds its layout first (`DatasetConcatCollection.version_identity`).
        """
        parent = self._parent_dataset
        if parent is not None and parent._answering_instance(flag) is self:
            return parent.version_identity(flag)
        return ((type(self).__qualname__, self.flag_index.get_spec(flag).version),)

    def _class_path(self) -> str:
        return f'{type(self).__module__}.{type(self).__qualname__}'

    @staticmethod
    def _call_config(config: dict | DatasetConfigType | None, enclosing: FetchFrame | None) -> Any:
        if config is None:
            return {}
        if enclosing is not None and config is enclosing.watched_config:
            return INHERIT
        from mesqual.datasets.dataset_config import DatasetConfig

        return dict(DatasetConfig._explicit_items(config))

    def _new_record(
        self,
        flag: FlagType,
        call_config: Any,
        call_kwargs: dict[str, Any],
        params: dict[str, Any],
        param_default_reads: dict[str, Any] | None = None,
    ) -> FetchRecord:
        return FetchRecord(
            dataset_class=self._class_path(),
            dataset_name=self.name,
            flag=str(flag),
            call_kwargs=snapshot(dict(call_kwargs)),
            call_config=call_config if call_config is INHERIT else snapshot(call_config),
            params=snapshot(params),
            version=self.version_identity(flag),
            param_default_reads=snapshot(param_default_reads or {}),
        )

    @property
    def _flag_param_names(self) -> set[str]:
        """Names of the flag parameters of the own specs: of the own flag index, else of `flag_specs`.

        A dataset with a parent has no index of its own, so the check reads
        its `flag_specs`, not the index of the parent.
        """
        if self._flag_index is not None:
            return self._flag_index.parameter_names
        return {d.name for spec in self.flag_specs for d in spec.params}

    @property
    def _accepted_dataset_params(self) -> dict[str, Dimension]:
        """The dataset parameters that `fetch` accepts, by name: the `fetch_params` of this class."""
        return {d.name: d for d in self.fetch_params}

    def _check_param_name_clash(self) -> None:
        """Raise `ValueError` if a name is a flag parameter and a dataset parameter of this dataset."""
        self._raise_on_param_name_clash(self._flag_param_names, set(self._accepted_dataset_params))

    def _raise_on_param_name_clash(self, flag_names: set[str], dataset_names: set[str]) -> None:
        """Raise `ValueError` if a name is in `flag_names` and in `dataset_names`."""
        clash = flag_names & dataset_names
        if clash:
            raise ValueError(
                f'Dataset {self.name}: {sorted(clash)} are flag parameters (FlagSpec.params) and dataset '
                f'parameters (fetch_params). A parameter name must have one owner. Rename one of them.'
            )

    @staticmethod
    def _given(**kwargs) -> dict[str, Any]:
        """`kwargs` without the `UNSET` values: the arguments that a typed `fetch` override got from the caller."""
        return {name: value for name, value in kwargs.items() if value is not UNSET}

    def _kwargs_for(self, flag: FlagType, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The parameters of `kwargs` that the spec of `flag` or this dataset declares."""
        declared = {d.name for d in self.flag_index.get_spec(flag).params} | set(self._accepted_dataset_params)
        return {k: v for k, v in kwargs.items() if k in declared}

    def _spec_of_accepted_flag(self, flag: FlagType) -> FlagSpec:
        spec = self.flag_index.get_spec(flag)
        if not self.flag_is_accepted(flag):
            raise ValueError(f'Flag {flag} not accepted by Dataset "{self.name}" of type {type(self)}.')
        return spec

    def _answering_instance(self, flag: FlagType) -> Dataset | None:
        """The child that answers `flag`, or `None` for a dataset without children that answer flags."""
        return None

    def _check_param_default_names(self, config: dict | DatasetConfigType | None) -> None:
        """Raise `ValueError` for a `param_defaults` name that no parameter of this dataset declares.

        Names of the instance config and of the call `config` must be flag
        parameters of the flag index or dataset parameters of this dataset
        (for a collection, also of its children). Names of a class config
        may also be dataset parameters of any `Dataset` subclass, so one
        class config can serve all datasets.
        """
        from mesqual.datasets.dataset_config import DatasetConfigManager

        own = self.flag_index.parameter_names | set(self._accepted_dataset_params)
        class_layers = [layer for _, layer in DatasetConfigManager.get_layers(type(self))[1:]]
        strict_names = set().union(*map(self._param_default_names, (self._config, config)))
        loose_names = set().union(*map(self._param_default_names, class_layers))
        unknown = (strict_names - own) | (loose_names - own - self._fetch_param_names_of_all_classes)
        if unknown:
            raise ValueError(
                f'Dataset {self.name}: param_defaults names parameters that this dataset does not declare: '
                f'{sorted(unknown)}. Flag parameters: {sorted(self.flag_index.parameter_names)}. '
                f'Dataset parameters: {sorted(self._accepted_dataset_params)}.'
            )

    @staticmethod
    def _param_default_names(config: dict | DatasetConfigType | None) -> set[str]:
        """Names in the explicit `param_defaults` of one config layer."""
        from mesqual.datasets.dataset_config import DatasetConfig

        param_defaults = dict(DatasetConfig._explicit_items(config)).get('param_defaults')
        return set(param_defaults) if isinstance(param_defaults, dict) else set()

    def _params_to_resolve(self, spec: FlagSpec) -> tuple[Dimension, ...]:
        """The parameters that this dataset resolves: the flag parameters (unless a collection) and `fetch_params`.

        Raises:
            ValueError: A flag parameter of `spec` has the name of a dataset
                parameter, for example of a spec added to the flag index
                after construction.
        """
        clash = {d.name for d in spec.params} & {d.name for d in self.fetch_params}
        if clash:
            raise ValueError(
                f'Spec {spec.name!r}: {sorted(clash)} are flag parameters of the spec and dataset parameters '
                f'(fetch_params) of {type(self).__name__}. A parameter name must have one owner. Rename one of them.'
            )
        flag_params = spec.params if self._resolves_flag_params else ()
        return (*flag_params, *self.fetch_params)

    def _resolve_params(
        self,
        spec: FlagSpec,
        flag: FlagType,
        call_kwargs: dict[str, Any],
        effective_config: DatasetConfigType,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve the parameters of a call.

        Each parameter that this dataset resolves gets the keyword argument,
        else the `param_defaults` entry, else the default of its dimension
        (`Dimension.resolve`). The values of dimensions that are resolved per
        dataset are checked later, in the frame of the fetch.
        The other keyword arguments pass through unchanged.

        Returns:
            The resolved parameters, and the `param_defaults` entries that
                the resolution used.
        """
        accepted = {d.name for d in spec.params} | set(self._accepted_dataset_params)
        unknown = set(call_kwargs) - accepted
        if unknown:
            raise ValueError(
                f'Flag {flag}: no flag spec or dataset class declares the parameters {sorted(unknown)}. '
                f'Declared: {sorted(accepted)}.'
            )
        defaults = effective_config.param_defaults or {}
        dimensions = self._params_to_resolve(spec)
        reads = {d.name: defaults[d.name] for d in dimensions if d.name not in call_kwargs and d.name in defaults}
        given = {**reads, **call_kwargs}
        params = {}
        for dimension in dimensions:
            if not dimension.has_default and dimension.name not in given:
                raise ValueError(
                    f'Flag {flag}: parameter {dimension.name!r} is required. Pass it to fetch or set param_defaults.'
                )
            try:
                params[dimension.name] = dimension.resolve(given)
            except ValueError as error:
                raise ValueError(f'Flag {flag}: {error}') from error
        params.update({name: value for name, value in call_kwargs.items() if name not in params})
        return params, reads

    def _check_dataset_values(self, spec: FlagSpec, flag: FlagType, params: dict[str, Any]) -> None:
        """Dataset-specific parameter checks: the variant and parameter values whose allowed values come from the dataset.

        A subclass adds its own checks, for example of its dataset
        parameters. It runs inside the frame of the fetch, so the fetches of the values
        functions are recorded as dependencies. Each dataset in a chain of
        routed calls checks the values again with the root dataset.
        """
        root = self._root_dataset()
        try:
            if spec.is_family and self._resolves_flag_params:
                spec.check_variant_values(flag, root)
            for dimension in self._params_to_resolve(spec):
                if dimension.is_resolved_per_dataset:
                    dimension.check_value(params[dimension.name], root)
        except FlagUnavailableError as error:
            raise FlagUnavailableError(str(flag), self.name, error.reason) from error
        except ValueError as error:
            raise ValueError(f'Flag {flag}: {error}') from error

    def _root_dataset(self) -> Dataset:
        dataset = self
        while dataset._parent_dataset is not None:
            dataset = dataset._parent_dataset
        return dataset

    def _fetch_from_cache_or_source(
        self, flag: FlagType, effective_config: DatasetConfigType, frame: FetchFrame, **kwargs
    ) -> pd.Series | pd.DataFrame:
        """Return the cached value if the entry of the cache key is up to date, else call `_fetch` and store the result.

        `_fetch` gets `_cache_config(effective_config)`, so the settings that
        apply after the cache never reach it. An entry is stored only if its
        record is up to date right after the fetch, so the cache never
        holds an entry that no later fetch can use. If the entry is not
        stored, or `Cache.set` raises, the fetch still returns the value
        and logs the reason at INFO, once per dataset class and flag. A stale
        entry logs its reason at DEBUG.
        """
        use_cache = self._cache is not None and effective_config.use_cache
        frame.stores = use_cache
        cache_config = self._cache_config(effective_config)
        if use_cache:
            key = CacheKey.of(frame.record)
            check = DependencyCheck(self, frame.record, cache_config)
            entry = self._up_to_date_entry(flag, key, check)
            if entry is not None:
                stored, value = entry
                frame.add_stored(stored)
                return value.copy()

        watched_config = FetchContext.watch(cache_config, frame)
        raw_data = self._fetch(flag, watched_config, **kwargs)
        processed_data = self._post_process_data(raw_data, flag, watched_config)

        if use_cache:
            record = frame.to_record()
            reason = check.stale_reason(record)
            if reason is None and not self._can_snapshot(record):
                reason = 'a recorded value cannot be deep-copied'
            if reason is None:
                try:
                    self._cache.set(key, processed_data, record)
                except Exception as error:
                    reason = f'the cache cannot store it ({type(error).__name__}: {error})'
            if reason is not None:
                _NotStoredLogs.log(self, flag, reason)

        return processed_data.copy()

    def _up_to_date_entry(self, flag: FlagType, key: CacheKey, check: DependencyCheck) -> Entry | None:
        """The entry of `key` if `check` finds it up to date, else `None`.

        A stale entry logs its reason at DEBUG. An exception of
        `Cache.entry` logs a warning and counts as a miss.
        """
        try:
            entry = self._cache.entry(key)
        except Exception as error:
            logger.warning(f'Flag {flag}: the cache cannot read the entry ({type(error).__name__}: {error}).')
            return None
        if entry is None:
            return None
        reason = check.stale_reason(entry[0])
        if reason is not None:
            logger.debug(f'Flag {flag}: cache entry is stale: {reason}')
            return None
        return entry

    @classmethod
    def _cache_config(cls, effective_config: DatasetConfigType) -> DatasetConfigType:
        """The effective config without the settings that apply after the cache. `_fetch` gets this config.

        The fields in `_AFTER_CACHE_FIELDS` (`time_index_filter` and
        `granularity`) are `None`. A read of the whole config records them
        as `None`, so neither setting makes an entry stale.
        """
        cache_config = copy.copy(effective_config)
        for name in cls._AFTER_CACHE_FIELDS:
            object.__setattr__(cache_config, name, None)
        return cache_config

    @staticmethod
    def _can_snapshot(record: FetchRecord) -> bool:
        """`True` if the record can be deep-copied. A value that `snapshot` kept as is cannot be checked later."""
        try:
            copy.deepcopy(record)
            return True
        except Exception:
            return False

    def _rebuild_call(
        self, stored: FetchRecord, parent_config: DatasetConfigType
    ) -> tuple[FetchRecord, DatasetConfigType]:
        """Rebuild the call of a stored dependency record with the current configs, for `DependencyCheck`.

        The config of the call is the current config of this dataset, then
        the recorded `call_config`, or `parent_config` for `INHERIT`. The
        parameters resolve from the recorded `call_kwargs` with that config.

        Returns:
            The record of the call as it would start now, and the config
                that its `_fetch` would get.
        """
        with FetchContext.internal():
            config = parent_config if stored.call_config is INHERIT else stored.call_config
            spec = self.flag_index.get_spec(stored.flag)
            record, effective_config, *_ = self._start_call(
                spec,
                stored.flag,
                config,
                stored.call_config,
                dict(stored.call_kwargs),
            )
            return record, self._cache_config(effective_config)

    def _child_datasets(self) -> tuple[Dataset, ...]:
        """All datasets that this dataset holds and fetches from: for a collection its children, also the interpreters of a platform dataset.

        Use it to walk the whole dataset tree, as `DependencyCheck` does.
        `_describe_children` differs: it hides the interpreters of a
        platform dataset, because the describe methods show a platform
        dataset as one dataset.
        """
        return ()

    @staticmethod
    def _with_granularity(config: dict | DatasetConfigType | None, granularity: Any) -> dict:
        """The fetch-time config `config` with `granularity` on top, for the `granularity` argument of `fetch`."""
        from mesqual.datasets.dataset_config import DatasetConfig

        return {**dict(DatasetConfig._explicit_items(config)), 'granularity': granularity}

    @staticmethod
    def _check_granularity(granularity: Any) -> None:
        """Raise `ValueError` if `granularity` is not `None` and not a positive fixed frequency (`parse_frequency`)."""
        if granularity is None:
            return
        from mesqual.energy_data_handling.granularity_converter import TimeSeriesGranularityConverter

        try:
            TimeSeriesGranularityConverter.parse_frequency(granularity)
        except ValueError as error:
            raise ValueError(f'granularity: {error}') from error

    def _convert_granularity(
        self,
        data: pd.Series | pd.DataFrame,
        flag: FlagType,
        granularity: str | pd.Timedelta | None,
    ) -> pd.Series | pd.DataFrame:
        """Convert `data` to `granularity` with the quantity type of the flag. A row `MultiIndex` converts on its time level."""
        if granularity is None:
            return data
        from mesqual.energy_data_handling.granularity_converter import TimeSeriesGranularityConverter

        time_level = self._time_level_of(data) if isinstance(data.index, pd.MultiIndex) else None
        quantity_type = self.flag_index.get_quantity_type_enum(flag)
        return TimeSeriesGranularityConverter().convert(data, granularity, quantity_type, time_level=time_level)

    def _apply_time_index_filter(
        self, data: pd.Series | pd.DataFrame, effective_config: DatasetConfigType
    ) -> pd.Series | pd.DataFrame:
        filters = effective_config.time_index_filter
        if filters is None:
            return data
        if isinstance(filters, TimeIndexFilter):
            filters = (filters,)
        time_level = self._time_level_of(data)
        for time_index_filter in filters:
            data = time_index_filter.apply(data, time_level)
        return data

    @staticmethod
    def _time_level_of(data: pd.Series | pd.DataFrame) -> int | None:
        """Position of the time level of a result, used by a time index filter without `level`.

        The time level is the first `DatetimeIndex` level of the row index.
        A result without such a level has no time level.
        """
        index = data.index
        if isinstance(index, pd.MultiIndex):
            for position, level in enumerate(index.levels):
                if isinstance(level, pd.DatetimeIndex):
                    return position
            return None
        return 0 if isinstance(index, pd.DatetimeIndex) else None

    def _warn_about_unknown_config_fields(self, effective_config: DatasetConfigType) -> None:
        if self._warns_about_unknown_config_fields:
            _UnknownConfigFieldWarnings.warn(self.name, effective_config, self._config_fields_of_parents())

    def _config_fields_of_parents(self) -> frozenset[str]:
        parent = self._parent_dataset
        if parent is None:
            return frozenset()
        return parent.get_config_type()._field_names() | parent._config_fields_of_parents()

    @property
    def accepted_flags(self) -> set[FlagType]:
        """
        Set of the flags of the specs of this dataset.

        It holds concrete flag names, computed from the declarations in
        `flag_specs` without opening source data. Flags of a flag family are
        not listed, because their values come from the data of the dataset.
        `flag_is_accepted` accepts them by their name template. Whether a
        source has the data of a flag is a separate question (availability).

        A subclass may narrow the spec flags using declarations of siblings
        or attributes, never source data.

        Returns:
            Set of flags that can be used with the fetch() method

        Examples:
            >>> 'Line.Results.flow_sent_net' in dataset.accepted_flags
            True
        """
        return {flag for spec in self.flag_specs if not spec.is_family for flag in spec.expand()}

    def get_accepted_flags_containing_x(self, x: str, match_case: bool = False) -> set[FlagType]:
        """
        Find all accepted flags containing a specific substring.

        Useful for discovering related data flags or filtering flags by category.

        Args:
            x: Substring to search for in flag names
            match_case: If True, performs case-sensitive search. Default is False.

        Returns:
            Set of accepted flags containing the substring

        Examples:
            >>> sorted(dataset.get_accepted_flags_containing_x('line.results.flow_sent'))
            ['Line.Results.flow_sent_down', 'Line.Results.flow_sent_net', 'Line.Results.flow_sent_up']
            >>> dataset.get_accepted_flags_containing_x('line.results', match_case=True)
            set()
        """
        return {f for f in self.accepted_flags if FlagDescriber.name_contains(f, x, match_case)}

    def flag_is_accepted(self, flag: FlagType) -> bool:
        """
        Boolean check whether a flag is accepted by the Dataset.

        `True` for a flag in `accepted_flags` and for a flag of a flag family
        spec of this dataset. Uses only declarations, never source data.
        """
        if flag in self.accepted_flags:
            return True
        return any(spec.is_family and spec.covers(flag) for spec in self.flag_specs)

    @property
    def flag_index(self) -> FlagIndex:
        """
        The flag index: lookup from a flag to its spec.

        Built from `flag_specs` at construction. A dataset constructed with a
        parent dataset, for example an interpreter, uses the index of its
        parent. Collections and platform datasets document their own rule.
        """
        if self._flag_index is None and self._parent_dataset is not None:
            return self._parent_dataset.flag_index
        return self._flag_index

    def _build_flag_index(self) -> FlagIndex | None:
        """Return an index of `flag_specs`, or `None` for a dataset with a parent: it uses the index of the parent."""
        if self._parent_dataset is not None:
            return None
        return FlagIndex(self.flag_specs)

    def _result_schema(self, flag: FlagType) -> FrameSchema:
        """The schema of the result of a fetch of `flag`. A collection transforms the schema of its children."""
        return self.flag_index.get_schema(flag)

    @property
    def cache(self) -> Cache | None:
        """The cache of fetch results of this dataset, or `None`.

        `fetch` uses it when `use_cache` is set (see
        [Cache](../architecture/other-contracts.md#cache)).
        """
        return self._cache

    def add_kpis_from_definitions(self, kpi_definitions: KPIDefinition | list[KPIDefinition]):
        """
        Generate and add KPIs from one or more KPI definitions.

        KPI definitions are templates that generate concrete KPI instances
        based on the dataset's structure. This method processes definitions
        and adds the resulting KPIs to the dataset's KPI collection.

        Args:
            kpi_definitions: Single KPIDefinition or list of definitions.
                Each definition's generate_kpis() method is called with
                this dataset to produce KPI instances.

        Examples:
            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> definitions = FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
            >>> dataset.add_kpis_from_definitions(definitions)
            >>> len(dataset.kpi_collection)
            2

        See Also:
            - [`add_kpi()`][mesqual.datasets.Dataset.add_kpi]: add one KPI.
            - [`add_kpis()`][mesqual.datasets.Dataset.add_kpis]: add several KPIs.
        """
        from mesqual.kpis.definitions.base import KPIDefinition

        if isinstance(kpi_definitions, KPIDefinition):
            kpis = kpi_definitions.generate_kpis(self)
            self.add_kpis(kpis)
        else:
            for kpi_def in kpi_definitions:
                kpis = kpi_def.generate_kpis(self)
                self.add_kpis(kpis)

    def add_kpis(self, kpis: Iterable[KPI]):
        """
        Add multiple KPIs to this dataset's KPI collection.

        Args:
            kpis: Iterable of KPI instances, factories, or classes to add
        """
        duplicates = []
        for kpi in kpis:
            if kpi in self.kpi_collection:
                duplicates.append(kpi)
            else:
                self.add_kpi(kpi)
        if duplicates:
            _num_duplicates = len(duplicates)
            logger.warning(
                f'{_num_duplicates} duplicates found and not added again or overwritten in {self.name}. ({duplicates[:3]}...)'
            )

    def add_kpi(self, kpi: KPI):
        """
        Add a single KPI to this dataset's KPI collection.

        Args:
            kpi: KPI instance, factory, or class to add
        """
        self.kpi_collection.add(kpi)

    def clear_kpi_collection(self):
        """Clear the KPI collection."""
        from mesqual.kpis.collection import KPICollection

        self.kpi_collection = KPICollection()

    @property
    def attributes(self) -> dict:
        """
        Access the metadata attributes dictionary for this dataset.

        Attributes store scenario-level metadata such as configuration parameters,
        simulation settings, or descriptive labels. These are useful for filtering,
        grouping, and annotating datasets in collections.

        Returns:
            dict: Dictionary of attribute key-value pairs. Keys are strings,
                values are primitive types (bool, int, float, str).

        Examples:
            >>> dataset.attributes
            {'first_date': '2024-03-28', 'last_date': '2024-04-01', 'days': 5, 'demand_factor': 1.0, 'periods': 2}

        See Also:
            - [`set_attributes()`][mesqual.datasets.Dataset.set_attributes] - Set attribute values
            - [`get_attributes_series()`][mesqual.datasets.Dataset.get_attributes_series] - Convert to pandas Series
        """
        return self._attributes

    def get_attributes_series(self) -> pd.Series:
        """
        Convert dataset attributes to a pandas Series.

        Creates a Series with attribute names as the index and attribute
        values as data. The Series name is set to the dataset name, making
        it suitable for concatenation with other datasets' attribute series.

        Returns:
            pd.Series: Series containing attribute values, indexed by attribute
                names, with the dataset name as the Series name

        Examples:
            >>> series = dataset.get_attributes_series()
            >>> series.name, series['days']
            ('base', 5)

        See Also:
            - [`attributes`][mesqual.datasets.Dataset.attributes] - Access raw attributes dictionary
            - [`set_attributes()`][mesqual.datasets.Dataset.set_attributes] - Set attribute values
        """
        att_series = pd.Series(self.attributes, name=self.name)
        return att_series

    def set_attributes(self, **kwargs: bool | int | float | str) -> None:
        """
        Set one or more metadata attributes on this dataset.

        Attributes are key-value pairs that store scenario metadata. They must
        use string keys and primitive values (bool, int, float, str) to ensure
        serializability and consistent comparison behavior.

        Args:
            **kwargs: Attribute key-value pairs to set. Keys must be strings,
                values must be bool, int, float, or str.

        Raises:
            TypeError: If any key is not a string
            TypeError: If any value is not bool, int, float, or str

        Examples:
            >>> dataset.set_attributes(year=2030, includes_nuclear=True)
            >>> dataset.attributes['year']
            2030

        See Also:
            - [`attributes`][mesqual.datasets.Dataset.attributes] - Access attributes dictionary
            - [`get_attributes_series()`][mesqual.datasets.Dataset.get_attributes_series] - Convert to pandas Series
        """
        self._add_attributes(kwargs)

    def _add_attributes(self, attributes: dict) -> None:
        for key, value in attributes.items():
            if not isinstance(key, str):
                raise TypeError(f'Attribute keys must be of type str. Your key {key} is of type {type(key)}.')
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError(
                    f'Attribute values must be of type (bool, int, float, str). '
                    f'Your value for {key} ({value}) is of type {type(value)}.'
                )
            self._attributes[key] = value

    @property
    def parent_dataset(self) -> 'DatasetLinkCollection':
        """The parent dataset of an interpreter: the platform dataset that holds it and its sibling interpreters.

        An interpreter answers a part of the flags of a platform dataset. It
        fetches the flags of its siblings through the parent, so it needs no
        reference to them. A fetch through the parent reaches an override of the
        sibling interpreter. The parent is a `DatasetLinkCollection`, usually a
        `PlatformDataset`.

        Returns:
            DatasetLinkCollection: The parent collection.

        Raises:
            RuntimeError: The parent is not set, for example for an interpreter
                used without a parent dataset.

        Examples:
            An interpreter that sums the generation per node from two flags of its siblings:

            ```python
            class NodeGeneration(MyInterpreter):
                flag_specs = (NODE_GENERATION_SPEC,)

                def _fetch(self, flag, effective_config, **params):
                    generation = self.parent_dataset.fetch('Generator.Results.generation', effective_config)
                    node_of_generator = self.parent_dataset.fetch('Generator.Model', effective_config)['node']
                    return generation.T.groupby(node_of_generator).sum().T
            ```

        See Also:
            - [`DatasetLinkCollection.get_dataset_by_type()`][mesqual.datasets.DatasetLinkCollection.get_dataset_by_type]:
              a sibling interpreter by its class.
        """
        if self._parent_dataset is None:
            raise RuntimeError(f'Parent dataset called without / before assignment.')
        return self._parent_dataset

    @parent_dataset.setter
    def parent_dataset(self, parent_dataset: 'DatasetLinkCollection'):
        """
        Set the parent collection for this dataset.

        Args:
            parent_dataset: The DatasetLinkCollection that will contain this dataset

        Raises:
            TypeError: If parent_dataset is not a DatasetLinkCollection instance
        """
        from mesqual.datasets.dataset_collection import DatasetLinkCollection

        if not isinstance(parent_dataset, DatasetLinkCollection):
            raise TypeError(f'Parent parent_dataset must be of type {DatasetLinkCollection.__name__}')
        self._parent_dataset = parent_dataset

    def _post_process_data(
        self, data: pd.Series | pd.DataFrame, flag: FlagType, config: DatasetConfigType
    ) -> pd.Series | pd.DataFrame:
        """Apply the standard post-processing to the result of `_fetch`.

        Handles duplicate index entries and sorts the row index if
        `auto_sort_index` is set. A model flag with duplicate ids raises,
        independent of the config. For other flags, `keep_duplicate_indices`
        decides which rows stay.

        Args:
            data: Raw data from `_fetch`.
            flag: The fetched flag.
            config: Effective config of the fetch.

        Returns:
            The post-processed data.

        Raises:
            ValueError: A model flag has duplicate ids, or
                `keep_duplicate_indices` is not `'first'`, `'last'` or `'all'`.

        Note:
            `fetch` calls this method. A subclass can override it to add
            post-processing and call `super()`.
        """
        if self.flag_index.get_item_type(flag) is ItemTypeEnum.Model:
            self._raise_on_duplicate_ids(data, flag)
        else:
            data = self._drop_duplicate_indices(data, flag, config.keep_duplicate_indices)
        if config.auto_sort_index:
            data = self._sort_index(data)
        return data

    def _sort_index(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Sort the row index of a result. `auto_sort_index` turns it on."""
        return data.sort_index()

    def _raise_on_duplicate_ids(self, data: pd.Series | pd.DataFrame, flag: FlagType) -> None:
        duplicated = data.index[data.index.duplicated()].unique()
        if len(duplicated):
            raise ValueError(
                f'Dataset {self.name}: model flag {flag} has duplicate ids {list(duplicated[:5])}. '
                f'A model flag has one row per object.'
            )

    def _drop_duplicate_indices(
        self,
        data: pd.Series | pd.DataFrame,
        flag: FlagType,
        keep: str,
    ) -> pd.Series | pd.DataFrame:
        if keep == 'all':
            return data
        if keep not in ('first', 'last'):
            raise ValueError(f"keep_duplicate_indices must be 'first', 'last' or 'all', got {keep!r}.")
        duplicated = data.index.duplicated(keep=keep)
        if duplicated.any():
            _DuplicateIndexWarnings.warn(self, flag, int(duplicated.sum()), keep)
            data = data.loc[~duplicated]
        return data

    def _prepare_config(self, config: dict | DatasetConfigType = None) -> DatasetConfigType:
        """
        Prepare the effective configuration for a fetch operation.

        Resolves the final configuration by merging the provided config override
        with the dataset's instance config (which already includes class-level
        and base defaults via DatasetConfigManager).

        Args:
            config: Optional override configuration. Can be:
                - None: Use instance config as-is
                - dict: Merge its keys
                - DatasetConfig: Merge directly with instance config

        Returns:
            The fully resolved configuration for the fetch operation.

        Raises:
            TypeError: If config is neither None, dict, nor DatasetConfig.
        """
        if config is None:
            return self.instance_config
        from mesqual.datasets.dataset_config import DatasetConfig

        if isinstance(config, (dict, DatasetConfig)):
            return self.instance_config.merge(config)
        raise TypeError(f'Config must be dict or {DatasetConfig.__name__}, got {type(config)}')

    @abstractmethod
    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        """
        Abstract method implementing the actual data retrieval logic.

        Subclasses must implement this method to define how data is retrieved
        for each flag. This is the core data access method that fetch() calls
        after configuration resolution and before post-processing.

        Args:
            flag: The validated flag to fetch data for
            effective_config: The fully resolved configuration for this operation
            **kwargs: Additional implementation-specific arguments

        Returns:
            DataFrame or Series containing the requested data. The returned
                data will be post-processed by _post_process_data() before being
                returned to the caller.

        Note:
            - This method should not perform flag validation (handled by fetch())
            - This method should not apply post-processing (handled separately)
            - This method should not handle caching (handled by fetch())
        """
        return pd.DataFrame()

    def fetch_multiple_flags_and_concat(
        self,
        flags: Iterable[FlagType],
        flag_concat_axis: int = 1,
        concat_level_name: str = 'variable',
        concat_level_at_top: bool = True,
        config: dict | DatasetConfigType = None,
        *,
        granularity: str | pd.Timedelta | None | Unset = UNSET,
        **kwargs: Any,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Fetch multiple flags and concatenate results into a single DataFrame.

        Convenience method for retrieving data from multiple flags and combining
        them into a single DataFrame with a MultiIndex. Useful for comparative
        analysis of multiple variables or creating wide-format data structures.

        All results must have the same shape of axes. Before the concat, each
        result is checked against the first one: both are a `Series` or both
        are a `DataFrame`, and each axis has the same number of levels and the
        same index type per level (for example `DatetimeIndex`). A `Series`
        has only its index to check.

        Args:
            flags: Iterable of flags to fetch and concatenate
            flag_concat_axis: Axis along which to concatenate (0=rows,
                1=columns). Default is 1 (columns). It is not the
                `concat_axis` of a concat collection, which is fixed when
                the collection is built.
            concat_level_name: Name for the new MultiIndex level identifying
                the source flag. Default is 'variable'.
            concat_level_at_top: If True, the flag level is the outermost level
                in the MultiIndex. If False, it's moved to the innermost level.
                Default is True.
            config: Optional configuration override (see fetch() for details)
            granularity: Target time resolution, as in `fetch`.
            **kwargs: Parameters. Each fetch gets the flag parameters that
                its spec declares, and the dataset parameters of this dataset.

        Returns:
            DataFrame with concatenated data and a MultiIndex identifying the
                source flag for each section

        Raises:
            ValueError: If two results differ in kind, number of levels or index
                type on an axis. The message names both flags.
            ValueError: A keyword argument is no flag parameter of any of the
                flags and no dataset parameter of this dataset.

        Examples:
            >>> demand_and_gas = dataset.fetch_multiple_flags_and_concat(
            ...     ['Node.Results.demand_total', 'Node.Results.generation_gas'], concat_level_name='variable',
            ... )
            >>> list(demand_and_gas.columns.names)
            ['variable', 'Node']

        See Also:
            - [`fetch()`][mesqual.datasets.Dataset.fetch] - Single flag data retrieval
        """
        flags = list(flags)
        self._check_declared_by_any(flags, kwargs)
        dfs = {
            str(flag): self.fetch(flag, config, granularity=granularity, **self._kwargs_for(flag, kwargs))
            for flag in flags
        }
        self._check_same_axes(dfs)
        df = pd.concat(
            dfs,
            axis=flag_concat_axis,
            names=[concat_level_name],
        )
        if not concat_level_at_top:
            df = move_top_level_last(df, flag_concat_axis)
        return df

    def _check_declared_by_any(self, flags: list[FlagType], kwargs: dict[str, Any]) -> None:
        declared = set(self._accepted_dataset_params).union(
            *({d.name for d in self.flag_index.get_spec(flag).params} for flag in flags)
        )
        unknown = set(kwargs) - declared
        if unknown:
            raise ValueError(
                f'No flag spec of {[str(f) for f in flags]} and no dataset class declares the parameters '
                f'{sorted(unknown)}. Declared: {sorted(declared)}.'
            )

    @classmethod
    def _check_same_axes(cls, dfs: dict[str, pd.Series | pd.DataFrame]) -> None:
        if not dfs:
            return
        (first_flag, first), *others = dfs.items()
        for flag, other in others:
            difference = cls._axes_difference(first, other)
            if difference:
                raise ValueError(f'Cannot concat the flags {first_flag!r} and {flag!r}: {difference}')

    @classmethod
    def _axes_difference(cls, first: pd.Series | pd.DataFrame, other: pd.Series | pd.DataFrame) -> str | None:
        if type(first) is not type(other):
            return f'one is a {type(first).__name__}, the other a {type(other).__name__}.'
        for name, ax_first, ax_other in zip(('index', 'columns'), first.axes, other.axes):
            if ax_first.nlevels != ax_other.nlevels:
                return f'the {name} has {ax_first.nlevels} and {ax_other.nlevels} levels.'
            types_first, types_other = cls._level_types(ax_first), cls._level_types(ax_other)
            if types_first != types_other:
                return f'the {name} has the index types {types_first} and {types_other}.'
        return None

    @staticmethod
    def _level_types(ax: pd.Index) -> list[str]:
        levels = ax.levels if isinstance(ax, pd.MultiIndex) else [ax]
        return ['Index' if isinstance(level, pd.RangeIndex) else type(level).__name__ for level in levels]

    @classmethod
    def get_flag_type(cls) -> Type[FlagType]:
        """
        Get the flag type class for this dataset type.

        Returns the type used for data flags in this dataset class. Subclasses
        can override to specify a custom flag type for type checking and
        validation.

        Returns:
            Type[FlagType]: The flag type class (default: FlagTypeProtocol)

        Note:
            Override in subclasses that use custom flag types.
        """
        from mesqual.flag.flag import FlagTypeProtocol

        return FlagTypeProtocol

    @classmethod
    def get_config_type(cls) -> Type[DatasetConfigType]:
        """
        Get the configuration type class for this dataset type.

        Returns the DatasetConfig subclass used by this dataset. Platform
        interfaces typically override this to return their extended config
        class with platform-specific options.

        Returns:
            Type[DatasetConfigType]: The config type class (default: DatasetConfig)

        Examples:
            >>> from mesqual.testing import DummyDataset
            >>> DummyDataset.get_config_type()
            <class 'mesqual.testing.dataset.DummyConfig'>

        Note:
            Override in platform dataset subclasses to return platform-specific
            config types with additional options.
        """
        from mesqual.datasets.dataset_config import DatasetConfig

        return DatasetConfig

    @property
    def instance_config(self) -> DatasetConfigType:
        """
        Get the effective configuration for this dataset instance.

        Computes the merged configuration by combining:

        1. Base config defaults
        2. Class-level config (set via set_class_config)
        3. Instance-level config (passed to __init__ or set via set_instance_config)

        Later settings override earlier ones. This is the configuration used
        by fetch() unless overridden by a fetch-time config parameter.

        Returns:
            DatasetConfigType: The fully resolved configuration for this instance

        Examples:
            >>> dataset.instance_config.use_cache
            True

        See Also:
            - [`set_instance_config()`][mesqual.datasets.Dataset.set_instance_config] - Replace instance configuration
            - [`set_class_config()`][mesqual.datasets.Dataset.set_class_config] - Set class-level defaults
            - [`DatasetConfigManager`][mesqual.datasets.DatasetConfigManager] - Configuration management system
        """
        from mesqual.datasets.dataset_config import DatasetConfigManager

        return DatasetConfigManager.get_effective_config(self.__class__, self._config)

    def describe_config(self) -> pd.DataFrame:
        """Describe the config fields of this dataset, one row per field.

        Returns:
            DataFrame indexed by field name with the columns `type`,
                `default`, `value` (effective value without a fetch-time config),
                `layer` (the layer that set the value: `'default'`,
                `'class <Name>'` or `'instance'`) and `description`.

        Examples:
            >>> dataset.describe_config().loc['use_cache', ['value', 'layer']].tolist()
            [True, 'default']
        """
        from mesqual.datasets.dataset_config import DatasetConfigManager

        layers = DatasetConfigManager.get_layers(type(self), self._config)
        effective = self.instance_config
        rows = {}
        for field in dataclasses.fields(effective):
            setting_layer = 'default'
            for label, config in layers[1:]:
                if field.name in config.explicit_fields:
                    setting_layer = label
            rows[field.name] = {
                'type': field.type if isinstance(field.type, str) else getattr(field.type, '__name__', str(field.type)),
                'default': effective._field_default(field.name),
                'value': getattr(effective, field.name),
                'layer': setting_layer,
                'description': field.metadata.get('description', ''),
            }
        return pd.DataFrame.from_dict(rows, orient='index')

    def set_instance_config(self, config: DatasetConfigType) -> None:
        """
        Replace the instance-level configuration for this dataset.

        Sets the configuration that will be merged with class-level defaults
        to produce the effective configuration used by fetch().

        Args:
            config: New configuration object to use for this instance

        Examples:
            >>> from mesqual.datasets import DatasetConfig
            >>> dataset.set_instance_config(DatasetConfig(use_cache=False, auto_sort_index=False))

        See Also:
            - [`instance_config`][mesqual.datasets.Dataset.instance_config] - Get the effective configuration
            - [`set_instance_config_kwargs()`][mesqual.datasets.Dataset.set_instance_config_kwargs] - Update settings
            - [`set_class_config()`][mesqual.datasets.Dataset.set_class_config] - Set class-level defaults
        """
        self._config = config

    def merge_instance_config(self, config: DatasetConfigType) -> None:
        """Merge `config` on top of the instance config of this dataset.

        Uses `DatasetConfig.merge` on copies: a field that `config` sets
        explicitly wins, and the other explicit fields of the instance
        config stay. Without an instance config, a copy of `config` becomes
        the instance config. Later changes to `config` do not reach the
        dataset.

        Args:
            config: Upper layer for the instance config.

        Examples:
            >>> from mesqual.datasets import DatasetConfig
            >>> dataset.set_instance_config(DatasetConfig(use_cache=False))
            >>> dataset.merge_instance_config(DatasetConfig(granularity='1h'))
            >>> dataset.instance_config.use_cache, dataset.instance_config.granularity
            (False, '1h')

        See Also:
            - [`set_instance_config()`][mesqual.datasets.Dataset.set_instance_config] - Replace the config
        """
        upper = copy.deepcopy(config)
        self._config = upper if self._config is None else copy.deepcopy(self._config).merge(upper)

    def set_instance_config_kwargs(self, **kwargs: Any) -> None:
        """
        Update individual configuration settings on this instance.

        Modifies specific attributes of the existing instance configuration
        without replacing the entire config object. Useful for tweaking
        individual settings.

        Args:
            **kwargs: Configuration attribute names and values to set

        Examples:
            >>> dataset.set_instance_config_kwargs(use_cache=False)
            >>> dataset.instance_config.use_cache
            False

        Creates an instance config of `get_config_type()` first if the
        dataset has none.

        See Also:
            - [`set_instance_config()`][mesqual.datasets.Dataset.set_instance_config] - Replace entire configuration
            - [`instance_config`][mesqual.datasets.Dataset.instance_config] - Get the effective configuration
        """
        if self._config is None:
            self._config = self.get_config_type()()
        for key, value in kwargs.items():
            setattr(self._config, key, value)

    @classmethod
    def set_class_config(cls, config: DatasetConfigType) -> None:
        """
        Set the class-level configuration for all instances of this dataset type.

        Class-level configuration serves as the default for all instances of
        this class. Instance-level configuration (set via set_instance_config)
        can override these defaults.

        Args:
            config: Configuration object to use as class-level defaults

        Examples:
            >>> from mesqual.datasets import DatasetConfig
            >>> from mesqual.testing import DummyDataset
            >>> DummyDataset.set_class_config(DatasetConfig(use_cache=False))
            >>> study.scen.get_dataset('high_demand').instance_config.use_cache
            False

        Note:
            This affects all instances of the class, including existing ones
            that haven't overridden the setting at instance level.

        See Also:
            - [`set_instance_config()`][mesqual.datasets.Dataset.set_instance_config] - Override for specific instances
            - [`DatasetConfigManager`][mesqual.datasets.DatasetConfigManager] - Configuration management system
        """
        from mesqual.datasets.dataset_config import DatasetConfigManager

        DatasetConfigManager.set_class_config(cls, config)

    def is_flag_available(self, flag: FlagType) -> bool | None:
        """Return whether this dataset can answer a flag, without reading time series data.

        A flag that the dataset does not accept, or a flag of a family with
        a variant value that is not a value on the dataset, is not available
        (`False`). Else:

        - Platform dataset: the `_is_available` of the answering
          interpreter decides. If it returns `None`, the flag is unknown
          (`None`) until its first successful fetch from the platform
          dataset.
        - Collection: the union of the children (available in one child,
          else unknown in one child, else `False`). Comparison: the
          intersection of both children (`False` if one child is `False`,
          else unknown if one child is unknown, else available).
        - Any other dataset: its own `_is_available`. `None` counts as
          available.

        The interpreter of a layer flag implements `_is_available` with
        the availability of the flags that its fetch reads: the base flag,
        the base model or border model, and the `input_flags` of the
        aggregation. `availability` and `flag_info` give the reason.

        Returns:
            `True`, `False`, or `None` for unknown.
        """
        return self.availability(flag).available

    def availability(self, flag: FlagType) -> Availability:
        """Return whether this dataset can answer a flag, with the reason. Reads no time series data.

        The same rules as `is_flag_available`, which returns the
        `available` field of the result. An interpreter uses it on its
        parent dataset to combine the availability of the flags that it
        fetches (`Availability.of_inputs`).

        Examples:
            >>> from mesqual.testing import DummyDataset, DummyScenario
            >>> without_prices = DummyDataset(DummyScenario(missing_tables=('node_price',)))
            >>> without_prices.availability('Country.Results.price').reason
            "DummyAreaFlags._is_available: unavailable input Node.Results.price (DummyTimeSeriesInterpreter._is_available: table 'node_price' not in the source)"
        """
        return self._availability(flag)

    def _is_available(self, flag: FlagType) -> bool | Availability | None:
        """Hook: `True` or `False` when this dataset knows cheaply whether it can answer `flag`, `None` for unknown.

        An interpreter implements it with cheap checks, for example on the
        source inventory of its parent dataset
        (`self.parent_dataset.source_inventory`). It reads the inventory and
        does not fetch. Default: `None`.

        It may return an `Availability` instead of a bool, so the reason
        names the missing table or input. An `Availability(None, reason)`
        is unknown with that reason. Unlike `None`, it decides the
        availability on a platform dataset, so an earlier interpreter of
        the chain is not asked.

        Examples:
            ```python
            class PriceInterpreter(MyInterpreter):
                def _is_available(self, flag):
                    return 'prices.csv' in self.parent_dataset.source_inventory.files

            class VolumeInterpreter(MyInterpreter):
                def _is_available(self, flag):
                    found = 'volumes.csv' in self.parent_dataset.source_inventory.files
                    return Availability(found, 'volumes.csv in the source' if found else 'no volumes.csv')
            ```
        """
        return None

    def _availability(self, flag: FlagType) -> Availability:
        if not self.flag_is_accepted(flag):
            return Availability(False, f'{self.name} does not accept the flag.')
        return self._family_value_problem(flag) or self._checked_availability(flag)

    def _family_value_problem(self, flag: FlagType) -> Availability | None:
        """`Availability(False, reason)` if a family variant value of `flag` is not a value on this dataset.

        Also if the values of a family variant need an unavailable flag.
        """
        spec = self.flag_index.get_spec(flag)
        if not spec.is_family:
            return None
        try:
            spec.check_variant_values(flag, self._root_dataset())
        except FlagUnavailableError as error:
            return Availability(False, error.reason)
        except ValueError as error:
            return Availability(False, str(error))
        return None

    def _checked_availability(self, flag: FlagType) -> Availability:
        """Availability of a flag that this dataset accepts."""
        checked = self._availability_from_hook(self._is_available(flag), f'{type(self).__name__}._is_available')
        return checked or Availability(True, f'{type(self).__name__} has no availability check')

    @staticmethod
    def _availability_from_hook(checked: bool | Availability | None, source: str) -> Availability | None:
        """Turn the result of an `_is_available` hook into an `Availability`.

        `None` stays `None`. A bool gets `source` as reason. An `Availability`
        keeps its state and gets `source` before its reason.
        """
        if checked is None:
            return None
        if isinstance(checked, Availability):
            return Availability(checked.available, f'{source}: {checked.reason}', checked.datasets)
        return Availability(checked, source)

    def _raise_if_unavailable(self, flag: FlagType) -> None:
        """Raise `FlagUnavailableError` if the availability check says unavailable.

        A family flag whose values need an unavailable flag raises in
        `_check_dataset_values`, which resolves the values anyway.
        """
        availability = self._checked_availability(flag)
        if availability.available is False:
            raise FlagUnavailableError(str(flag), self.name, availability.reason)

    def _flag_availability_by_spec(self) -> dict[str, dict[str, Availability]]:
        """Per spec name: the availability of each flag of the spec. Flag families resolve with the dataset."""
        result = {}
        for spec in self.flag_index.specs:
            flags = spec.expand(self._root_dataset()) if spec.is_family else spec.expand()
            result[spec.name] = {flag: self._availability(flag) for flag in flags}
        return result

    def _values_dataset(self, flag: FlagType) -> Dataset:
        """The dataset that resolves the values functions of the dimensions of `flag`."""
        return self._root_dataset()

    def _flag_chain_names(self, flag: FlagType) -> tuple[str, ...]:
        """The interpreter chain of `flag`, for `flag_info`. Empty for a dataset that does not route flags."""
        return ()

    def _describe_children(self) -> list[Dataset]:
        """The children shown in `describe_datasets` and in the union of `describe_flags`."""
        return []

    @classmethod
    def _declared_flag_specs(cls) -> tuple[FlagSpec, ...]:
        """The specs that the class declares, for `describe_class_flags`."""
        return tuple(cls.flag_specs)

    @classmethod
    def _class_layers(cls) -> LayerSet:
        """The area and border layers registered on the class. A plain dataset class has none."""
        return LayerSet()

    @classmethod
    def _class_flag_index(cls) -> FlagIndex:
        """The flag index of the declared specs and registered layers of the class, without reference checks.

        It answers class-level questions, for example the layers of a layer
        spec in `describe_class_flags`.
        """
        return FlagIndex(cls._declared_flag_specs(), cls._class_layers())

    @classmethod
    def _describes_class_flags(cls) -> bool:
        """`False` for a collection class other than a leaf: it has no flags of its own."""
        return True

    @classmethod
    def describe_class_flags(
        cls,
        contains: str | None = None,
        layer: str | None = None,
        item_type: ItemTypeEnum | str | None = None,
        as_records: bool = False,
    ) -> pd.DataFrame | list[dict]:
        """Describe the flag specs that this class declares, one row per spec. Needs no source.

        On a platform dataset class: the specs of its answering
        interpreters. On a dataset or interpreter class: its specs.
        Flag families show their template, and dimensions with values per
        dataset show `'resolved per dataset'`.

        Args:
            contains: Keep specs whose name or one of whose flags contains
                this substring, case-insensitive.
            layer: Keep specs on this area or border layer.
            item_type: Keep specs of this item type.
            as_records: Return a list of plain dicts, ready for JSON.

        Returns:
            One row per spec, indexed by name, with the columns `item_type`
                (`Model`, `TimeSeries` or `Other`), `object` (the model flag; for
                a layer spec without `model_flag`, the model flags of its
                layers, in the order of `layers`), `unit`, `quantity_type`, `params` (flag and dataset parameters
                with their defaults), `description`, `variants` and `layers`.

        Raises:
            TypeError: Called on a collection class other than a platform
                dataset class, or through an instance of one. A collection
                has no flags of its own: use `describe_flags()` on the
                instance.

        See Also:
            - [`describe_flags`][mesqual.datasets.Dataset.describe_flags]: the flags of one dataset
              instance, with resolved flag families and layers, effective parameter defaults and
              availability. Same index, and the same columns plus some more.

        Examples:
            >>> from mesqual.testing import DummyDataset
            >>> DummyDataset.describe_class_flags(contains='loading')[['object', 'unit']]
                                      object      unit
            name
            Line.Results.loading  Line.Model  per_unit
        """
        rows = FlagDescriber(cls).class_spec_rows()
        rows = FlagDescriber.filter_rows(rows, contains, layer, item_type)
        return FlagDescriber.output(rows, as_records, hidden=('flags_list',))

    def describe_flags(
        self,
        contains: str | None = None,
        layer: str | None = None,
        item_type: ItemTypeEnum | str | None = None,
        available_only: bool = True,
        as_records: bool = False,
    ) -> pd.DataFrame | list[dict]:
        """Describe the flags of this dataset, one row per spec. Loads no time series data.

        As `describe_class_flags`, with the values of flag families and layers
        resolved for this dataset (from model flags), the effective parameter
        defaults with their source (`spec`, `dataset`, `config`, `required`,
        or `per dataset` on a collection whose children resolve different
        defaults), and the availability. A collection describes the union of
        its children.

        Args:
            contains: Keep specs whose name or one of whose flags contains
                this substring, case-insensitive.
            layer: Keep specs on this area or border layer.
            item_type: Keep specs of this item type.
            available_only: Drop the specs whose flags are all unavailable.
                Specs with unknown availability stay.
            as_records: Return a list of plain dicts, ready for JSON.

        Returns:
            One row per spec, indexed by name, with the columns of
                `describe_class_flags` in the same order, and `flags` (number of flags that are
                available or unknown), `availability` (`available`, `unknown`,
                `partial` if some flags are unavailable, or `unavailable`) and
                `datasets` (for a collection: the children that have the flag,
                if not all of them).

        See Also:
            - [`describe_class_flags`][mesqual.datasets.Dataset.describe_class_flags]: the specs as
              declared on the class, without a source; no resolved values and no availability.

        Examples:
            >>> dataset.describe_flags(contains='loading')[['object', 'availability']]
                                      object availability
            name
            Line.Results.loading  Line.Model    available
        """
        rows = FlagDescriber.filter_rows(FlagDescriber(self).flag_rows(), contains, layer, item_type)
        if available_only:
            rows = [row for row in rows if row['availability'] != 'unavailable']
        hidden = ('flags_list',) if self._describe_children() else ('flags_list', 'datasets')
        return FlagDescriber.output(rows, as_records, hidden=hidden)

    def flag_info(self, flag: FlagType) -> FlagInfo:
        """Return everything mesqual knows about one flag of this dataset, without its data.

        Returns:
            `FlagInfo` with the spec and its base, the aggregation, the
                variant values, the combined description, unit and quantity type
                with their source, the schema, the parameters with their
                effective defaults and source, `fill_value`, the model flag, the
                availability with its reason, the interpreter chain and the
                recorded dependencies of the last fetch without arguments.

        Raises:
            UnknownFlagError: No spec declares the flag.

        Examples:
            >>> dataset.flag_info('Line.Results.flow_received_net').description
            'Power flow on the line. Side: where the flow leaves the line, after losses. Direction: up minus down.'
        """
        return FlagDescriber(self).flag_info(flag)

    @classmethod
    def describe_class_area_layers(cls, as_records: bool = False) -> pd.DataFrame | list[dict]:
        """Describe the area layers registered on this class, one row per layer.

        Args:
            as_records: Return a list of plain dicts, ready for JSON.

        Returns:
            One row per area layer, indexed by name, with the columns
                `index`, `refers_to` (base model and layer model),
                `mapping_columns` (the membership column of the base model),
                `flags` (number of declared flag specs on the layer) and
                `description`. Empty for a class without area layers.

        Raises:
            TypeError: Called on a collection class other than a platform
                dataset class, or through an instance of one.

        See Also:
            - [`describe_class_border_layers`][mesqual.datasets.Dataset.describe_class_border_layers]: the border layers.
        """
        return FlagDescriber.output(
            FlagDescriber(cls).area_layer_rows(), as_records, columns=FlagDescriber.LAYER_COLUMNS
        )

    @classmethod
    def describe_class_border_layers(cls, as_records: bool = False) -> pd.DataFrame | list[dict]:
        """Describe the border layers registered on this class, one row per layer.

        Args:
            as_records: Return a list of plain dicts, ready for JSON.

        Returns:
            One row per border layer, indexed by name, with the columns
                `index`, `refers_to` (area layer and edge model),
                `mapping_columns` (the edge ends of the edge model), `flags`
                (number of declared flag specs on the layer) and `description`.
                Empty for a class without border layers.

        Raises:
            TypeError: Called on a collection class other than a platform
                dataset class, or through an instance of one.

        See Also:
            - [`describe_class_area_layers`][mesqual.datasets.Dataset.describe_class_area_layers]: the area layers.
        """
        return FlagDescriber.output(
            FlagDescriber(cls).border_layer_rows(), as_records, columns=FlagDescriber.LAYER_COLUMNS
        )

    def describe_datasets(
        self,
        max_depth: int | None = None,
        as_records: bool = False,
    ) -> pd.DataFrame | list[dict]:
        """Describe the tree of this dataset and its children, one row per dataset.

        A platform dataset is a leaf: its interpreters are not listed.

        Args:
            max_depth: Number of levels below this dataset. `None`: the full
                tree. `1`: this dataset and its direct children. `0`: this
                dataset only.
            as_records: Return a list of plain dicts, ready for JSON.

        Returns:
            One row per dataset, depth first, indexed by `path` (the names
                from this dataset down, joined with `/`), with the columns
                `name`, `class`, `description`, `attributes`, `config` (the
                explicit fields of the instance config) and `children` (all
                children, also below `max_depth`).

        Raises:
            ValueError: `max_depth` is negative.

        Examples:
            >>> study.scen.describe_datasets(max_depth=1)['class'].tolist()
            ['DatasetConcatCollection', 'DatasetMergeCollection', 'DummyDataset', 'DummyDataset', 'DummyDataset']
        """
        if max_depth is not None and max_depth < 0:
            raise ValueError(f'max_depth must be None or at least 0, got {max_depth}.')
        return FlagDescriber.output(DatasetTreeDescriber(self).rows(max_depth), as_records, index='path')

    def __str__(self) -> str:
        return self.name
