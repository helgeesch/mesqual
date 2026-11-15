from __future__ import annotations

import logging
import dataclasses
from collections import Counter
from functools import reduce
from typing import Any, ClassVar, Generic, Iterable, Literal, TYPE_CHECKING, Iterator, Sequence
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from mesqual.datasets.dataset import Dataset
from mesqual.datasets.describe import Availability
from mesqual.datasets.exceptions import AmbiguousOverrideError, FlagUnavailableError
from mesqual.datasets.fetch_context import FetchContext, FetchFrame
from mesqual.enums import ItemTypeEnum
from mesqual.flag.dimension import Dimension
from mesqual.flag.flag_index import FlagIndex
from mesqual.flag.schema import FrameSchema, Level
from mesqual.utils.pandas_utils.combine_df import combine_dfs
from mesqual.utils.pandas_utils.is_numeric import pd_is_numeric
from mesqual.utils.pandas_utils.move_level import move_top_level_last
from mesqual.utils.set_aggregations import nested_union
from mesqual.utils.intersect_dicts import get_intersection_of_dicts
from mesqual.typevars import DatasetType, DatasetConfigType, FlagType, FlagIndexType

if TYPE_CHECKING:
    from mesqual.kpis.definitions.base import KPIDefinition
    from mesqual.kpis.collection import KPICollection
    from mesqual.cache import Cache

logger = logging.getLogger(__name__)


class DatasetCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    Dataset[DatasetConfigType, FlagType, FlagIndexType],
    ABC,
):
    """
    Abstract base class for collections of datasets.

    A DatasetCollection is a Dataset with child datasets. It has the Dataset
    API, so a collection can be a child of another collection. It gives access
    to its children, accepts the flags of all children and runs KPI operations
    on all children. A collection can be used wherever a Dataset is expected.

    Each collection class gives the schema of its results with
    `transform_schema`, so `flag_info(flag).schema` describes the fetched
    result.

    Attributes:
        datasets (list[DatasetType]): List of child datasets in this collection
    """

    _warns_about_unknown_config_fields = False
    _resolves_flag_params = False
    _reads_source = False
    _is_leaf: ClassVar[bool] = False
    """`True`: the describe methods and availability treat this collection as one dataset, without children (a platform dataset)."""

    def __init__(
        self,
        datasets: list[DatasetType] = None,
        name: str = None,
        parent_dataset: Dataset = None,
        attributes: dict = None,
        cache: Cache = None,
        config: DatasetConfigType = None,
    ):
        """Create a collection of child datasets.

        Args:
            datasets: The child datasets. Their names must be unique.
            name: Name of the collection.
            parent_dataset: Optional parent dataset.
            attributes: Attributes of the collection.
            cache: Optional cache of fetch results.
            config: Instance config.

        Raises:
            ValueError: Two children have the same name, or two children
                have different specs with the same name.
        """
        self._union_index: FlagIndex | None = None
        self._union_source: tuple[int, ...] | None = None
        self.datasets: list[DatasetType] = list(datasets) if datasets else []
        super().__init__(
            name=name,
            parent_dataset=parent_dataset,
            attributes=attributes,
            cache=cache,
            config=config,
        )
        self._check_unique_names()

    def _check_unique_names(self) -> None:
        duplicates = sorted(name for name, count in Counter(ds.name for ds in self.datasets).items() if count > 1)
        if duplicates:
            raise ValueError(f'Collection {self.name}: the dataset names {duplicates} are not unique.')

    @property
    def dataset_iterator(self) -> Iterator[DatasetType]:
        for ds in self.datasets:
            yield ds

    @property
    def flag_index(self) -> FlagIndex:
        """The flag index: the union of the flag indexes of the children.

        Raises:
            ValueError: Two children have different specs with the same name.
        """
        if self._flag_index is not None:
            return self._flag_index
        return self._union_of_child_indexes()

    def _union_of_child_indexes(self) -> FlagIndex:
        """The union of the flag indexes of the children, built again when a child index changes."""
        child_indexes = tuple(ds.flag_index for ds in self.datasets)
        if self._union_source != tuple(map(id, child_indexes)):
            self._union_index = FlagIndex.union(child_indexes)
            self._union_source = tuple(map(id, child_indexes))
        return self._union_index

    def _build_flag_index(self) -> FlagIndex | None:
        """Return `None`: the index of a collection is the union of the indexes of its children."""
        return None

    def version_identity(self, flag: FlagType) -> tuple:
        """Return one entry per child that a fetch of `flag` can read from, in order.

        Each entry is a `(class path, name)` pair. A collection gives no
        version of its own data; its children record their own identities.
        The pairs make an entry of the collection stale for an added or
        removed child, a child of another class or name, or a flag that
        becomes available or unavailable on a child. A child replaced by one
        of the same class and name keeps the entry: the name identifies the
        source (see [Cache](../architecture/other-contracts.md#cache)).
        `DatasetConcatCollection.version_identity` adds the concat layout
        first.
        """
        return tuple((ds._class_path(), ds.name) for ds in self._fetchable_children(flag))

    def transform_schema(self, schema: FrameSchema) -> FrameSchema:
        """Return the schema of a result of this collection, from the schema of the result of a child.

        Link, merge, sum and comparison collections keep the schema. A
        concat collection adds its level.
        """
        return schema

    def _result_schema(self, flag: FlagType) -> FrameSchema:
        """The schema of the result of a fetch of `flag`: `transform_schema` of the schema of the first child."""
        if self._is_leaf:
            return super()._result_schema(flag)
        children = self._fetchable_children(flag)
        if not children:
            return super()._result_schema(flag)
        return self.transform_schema(children[0]._result_schema(flag))

    def _check_param_name_clash(self) -> None:
        """Run `_check_children` on the children."""
        self._check_children(self.datasets)

    def _check_children(self, children: list[Dataset]) -> None:
        """Raise `ValueError` if `children`, as the children of this collection, clash.

        The checks of the constructor and of `add_dataset`: a name is a flag
        parameter of one child and a dataset parameter of this collection or
        of a child, or two children have different specs with the same name.
        """
        self._raise_on_param_name_clash(self._flag_param_names_with(children), set(self._dataset_params_with(children)))
        # The children of a platform dataset share one index: read each index object once.
        indexes = {id(ds.flag_index): ds.flag_index for ds in children}.values()
        FlagIndex.union(indexes)

    @property
    def _flag_param_names(self) -> set[str]:
        """Names of the flag parameters of the own specs and of all children, collected recursively."""
        return self._flag_param_names_with(self.datasets)

    def _flag_param_names_with(self, children: Iterable[Dataset]) -> set[str]:
        """Names of the flag parameters of the own specs and of `children`, collected recursively."""
        return super()._flag_param_names.union(*(ds._flag_param_names for ds in children))

    @property
    def _accepted_dataset_params(self) -> dict[str, Dimension]:
        """The `fetch_params` of this class and the dataset parameters of all children, collected recursively.

        `fetch` takes the dataset parameters of the children and passes them
        on, for example `comparison_type` through a concat collection of
        comparisons. A parameter of this class wins over one of a child with
        the same name.
        """
        return self._dataset_params_with(self.datasets)

    def _dataset_params_with(self, children: Iterable[Dataset]) -> dict[str, Dimension]:
        """The `fetch_params` of this class and the dataset parameters of `children`, collected recursively."""
        accepted: dict[str, Dimension] = {}
        for ds in children:
            for name, dimension in ds._accepted_dataset_params.items():
                accepted.setdefault(name, dimension)
        return {**accepted, **super()._accepted_dataset_params}

    def _kwargs_for_child(self, ds: Dataset, flag: FlagType) -> dict:
        """The keyword arguments for the fetch of `flag` from the child `ds`. Call it only in `_fetch` of this collection.

        They come from the parameters of the open fetch of this collection
        that the call or the `param_defaults` of this collection gave
        (`FetchRecord.given_params`): the flag parameters of the spec, and
        the dataset parameters that `ds` accepts. So a value in the call and
        the same value in `param_defaults` reach the child in the same way,
        also a dataset parameter of this collection, for example
        `comparison_type` of the comparisons of a comparison of comparisons.
        A dataset parameter that resolved to the default of its dimension
        does not reach the child, so the child resolves it with its own
        config. A plain child in a mixed collection of scenarios and
        comparisons gets no comparison parameters. A `param_defaults` entry
        for a parameter that this collection does not resolve reaches the
        child through the effective config.

        Raises:
            RuntimeError: No fetch of this collection is open, or a nested
                fetch is open (see `_open_frame`).
        """
        flag_params = {d.name for d in self.flag_index.get_spec(flag).params}
        child_params = ds._accepted_dataset_params
        given = self._open_frame().read_given_params()
        return {k: v for k, v in given.items() if k in flag_params or k in child_params}

    def _open_frame(self) -> FetchFrame:
        """The frame of the fetch of this collection that is open, from `_fetch` of this collection.

        Raises:
            RuntimeError: The innermost open frame is not one of this
                collection: `_fetch` runs outside of `fetch`, or in a thread
                started without `contextvars.copy_context().run(...)`.
        """
        frame = FetchContext.current()
        if frame is None or frame.dataset is not self:
            raise RuntimeError(
                f'{type(self).__name__} {self.name}: no open fetch of this collection. '
                f'Call _kwargs_for_child only in _fetch, in the context of fetch.'
            )
        return frame

    @property
    def attributes(self) -> dict:
        child_dataset_atts = [ds.attributes for ds in self.datasets]
        attributes_that_all_childs_have_in_common = get_intersection_of_dicts(child_dataset_atts)
        return {**attributes_that_all_childs_have_in_common, **self._attributes.copy()}

    def get_merged_kpi_collection(self, deep: bool = True) -> KPICollection:
        """
        Merge KPI collections from all child datasets.

        This method collects KPIs from all child datasets' kpi_collection
        properties and returns a unified collection. Optionally recurses into
        nested DatasetCollections.

        Args:
            deep: If True, recursively merge from nested DatasetCollections

        Returns:
            KPICollection containing all KPIs from all child datasets

        Examples:
            >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
            >>> definitions = FlagAggKPIBuilder().for_flag('Country.Results.price').with_aggregation(Aggregations.Mean).build()
            >>> study.scen.add_kpis_from_definitions_to_all_child_datasets(definitions)
            >>> len(study.scen.get_merged_kpi_collection())  # 4 scenarios, 2 countries
            8
        """
        from mesqual.kpis.collection import KPICollection

        merged = KPICollection()

        for ds in self.datasets:
            # Add KPIs from this dataset
            merged.extend(ds.kpi_collection._kpis)

            # Recursively add from nested collections
            if deep and isinstance(ds, DatasetCollection):
                nested_merged = ds.get_merged_kpi_collection(deep=deep)
                merged.extend(nested_merged._kpis)

        return merged

    def add_kpis_from_definitions_to_all_child_datasets(self, kpi_definitions: KPIDefinition | list[KPIDefinition]):
        for ds in self.dataset_iterator:
            ds.add_kpis_from_definitions(kpi_definitions)

    def clear_kpi_collection_for_all_child_datasets(self, deep: bool = True):
        for ds in self.datasets:
            ds.clear_kpi_collection()
            if deep and isinstance(ds, DatasetCollection):
                ds.clear_kpi_collection_for_all_child_datasets(deep=deep)

    @abstractmethod
    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        pass

    def flag_is_accepted(self, flag: FlagType) -> bool:
        return any(ds.flag_is_accepted(flag) for ds in self.datasets)

    @staticmethod
    def _fetches_from(ds: Dataset, flag: FlagType) -> bool:
        """`True` if the child `ds` accepts `flag` and the flag is not unavailable there."""
        return ds.flag_is_accepted(flag) and ds.is_flag_available(flag) is not False

    def _fetchable_children(self, flag: FlagType) -> list[DatasetType]:
        """The children that accept `flag` and on which it is not unavailable, in order. Empty if there are none."""
        return [ds for ds in self.datasets if self._fetches_from(ds, flag)]

    def _children_to_fetch(self, flag: FlagType) -> list[DatasetType]:
        """The children that a fetch of `flag` reads from, in order.

        `fetch` raises `ValueError` before this for a flag that no child
        accepts.

        Raises:
            FlagUnavailableError: Children accept the flag, but it is unavailable on all of them.
        """
        fetchable = self._fetchable_children(flag)
        if not fetchable:
            raise FlagUnavailableError(str(flag), self.name, self.availability(flag).reason)
        return fetchable

    def _availability(self, flag: FlagType) -> Availability:
        if self._is_leaf:
            return super()._availability(flag)
        return self._combine_availability({ds.name: ds._availability(flag) for ds in self.datasets})

    _combine_availability = staticmethod(Availability.union)
    """Combine the availability of the children: the union (available in one child, else unknown in one)."""

    def _flag_availability_by_spec(self) -> dict[str, dict[str, Availability]]:
        """Union over the children: per spec and flag, the availability in the children that have the spec."""
        if self._is_leaf:
            return super()._flag_availability_by_spec()
        by_flag: dict[str, dict[str, dict[str, Availability]]] = {}
        missing = Availability(False, 'no flag of the spec')
        for ds in self.datasets:
            for spec_name, flags in ds._flag_availability_by_spec().items():
                for flag, availability in flags.items():
                    by_flag.setdefault(spec_name, {}).setdefault(flag, {})[ds.name] = availability
        return {
            spec_name: {
                flag: self._combine_availability({ds.name: children.get(ds.name, missing) for ds in self.datasets})
                for flag, children in flags.items()
            }
            for spec_name, flags in by_flag.items()
        }

    def _values_dataset(self, flag: FlagType) -> Dataset:
        """The values dataset of the first child that fetches `flag`."""
        if self._is_leaf:
            return super()._values_dataset(flag)
        children = self._fetchable_children(flag)
        return children[0]._values_dataset(flag) if children else self

    @classmethod
    def _describes_class_flags(cls) -> bool:
        return cls._is_leaf

    def _describe_children(self) -> list[Dataset]:
        return [] if self._is_leaf else list(self._child_datasets())

    def _child_datasets(self) -> tuple[Dataset, ...]:
        return tuple(self.datasets)

    @property
    def accepted_flags(self) -> set[FlagType]:
        return nested_union([ds.accepted_flags for ds in self.datasets])

    def get_dataset(self, key: str = None) -> DatasetType:
        if key is None:
            if not self.datasets:
                raise ValueError('No datasets available')
            return self.datasets[0]

        for ds in self.datasets:
            if ds.name == key:
                return ds

        raise KeyError(f"Dataset with name '{key}' not found")

    def add_datasets(self, datasets: Iterable[DatasetType]):
        for ds in datasets:
            self.add_dataset(ds)

    def add_dataset(self, dataset: DatasetType):
        """Add a child dataset. A child with the same name replaces the existing one and logs a warning.

        The collection runs the checks of the constructor on the children
        with the new child. If a check fails, the collection does not change.

        Raises:
            TypeError: The dataset is not of `get_child_dataset_type()`.
            ValueError: A name is a flag parameter of one dataset and a
                dataset parameter (`fetch_params`) of another in the
                collection, or the new child and another child have different
                specs with the same name.
        """
        if not self._accepts_child_type(type(dataset)):
            raise TypeError(f'Can only add data sets of type {self.get_child_dataset_type().__name__}.')
        children = {ds.name: ds for ds in self.datasets}
        replaced = dataset.name in children
        children[dataset.name] = dataset
        self._check_children(list(children.values()))
        if replaced:
            logger.warning(f'Dataset {self.name}: dataset {dataset.name} already in this collection. Replacing it.')
        self.datasets[:] = children.values()

    @classmethod
    def get_child_dataset_type(cls) -> type[DatasetType]:
        return Dataset

    @classmethod
    def _accepts_child_type(cls, dataset_type: type[Dataset]) -> bool:
        """`True` if a child of this type may join the collection: a subclass of `get_child_dataset_type()`."""
        return issubclass(dataset_type, cls.get_child_dataset_type())

    def fetch_merged(
        self,
        flag: FlagType,
        config: dict | DatasetConfigType = None,
        precedence: Literal['first', 'last'] | Sequence[str] = 'first',
        **kwargs: Any,
    ) -> pd.Series | pd.DataFrame:
        """Fetch a flag from all child datasets and merge the results, as `DatasetMergeCollection` does.

        The fetch goes through a temporary `DatasetMergeCollection` of the
        children. It has no cache, so it caches nothing, and its record is
        not kept. The records of the children live on the child datasets.
        Dataset parameters of the children (for example `comparison_type`)
        pass through to them.

        Args:
            flag: The flag.
            config: Fetch-time config, as in `fetch`.
            precedence: Which child wins where two children overlap, as in
                `DatasetMergeCollection`.
            **kwargs: Flag parameters and dataset parameters of the children.
        """
        temp_merge_collection = self.get_merged_dataset_collection(precedence)
        return temp_merge_collection.fetch(flag, config, **kwargs)

    def get_merged_dataset_collection(
        self,
        precedence: Literal['first', 'last'] | Sequence[str] = 'first',
    ) -> 'DatasetMergeCollection':
        """Return a `DatasetMergeCollection` of the children of this collection.

        Args:
            precedence: Which child wins where two children overlap, as in
                `DatasetMergeCollection`.
        """
        return DatasetMergeCollection(
            datasets=self.datasets,
            name=f'{self.name} merged',
            precedence=precedence,
        )


class DatasetLinkCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
):
    """
    Routes each flag to the one child dataset that accepts it.

    A link collection combines child datasets that each answer a different
    subset of flags into one dataset. It does not combine scenarios; use
    `DatasetConcatCollection` or `DatasetMergeCollection` for that.

    Routing:
        `fetch(flag)` goes to the one child that accepts the flag. If no
        child accepts the flag, or the flag is unavailable on that child,
        it raises `ValueError`. Two children that accept the same flag
        raise `AmbiguousOverrideError` at construction.

    Children reach their siblings through `parent_dataset`:

    - `self.parent_dataset.fetch('other_flag', effective_config)`
    - `self.parent_dataset.get_dataset_by_type(SomeDataset)`

    `PlatformDataset` builds on this class with an interpreter registry. To
    replace a platform interpreter, register an override with
    `PlatformDataset.register_interpreter(Override, override_interpreter=Original, version=...)`.

    The constructor checks the `model_flag` references of the union of the
    child specs. So a time series child may refer to the model spec of a
    sibling. Fetch such a child through the link collection: a child fetched
    directly checks its own flag index.

    Examples:
        ```python
        class ModelData(Dataset):
            flag_specs = (GENERATOR_MODEL_SPEC,)  # 'Generator.Model'
            ...

        class TimeSeriesData(Dataset):
            flag_specs = (GENERATION_SPEC,)  # 'Generator.Results.generation', model_flag='Generator.Model'
            ...

        linked = DatasetLinkCollection([ModelData(), TimeSeriesData()], name='base')
        linked.fetch('Generator.Results.generation')  # answered by TimeSeriesData
        ```

    See Also:
        - [`Dataset.parent_dataset`][mesqual.datasets.Dataset.parent_dataset]
        - [`PlatformDataset.register_interpreter`][mesqual.datasets.PlatformDataset.register_interpreter]
        - [`get_dataset_by_type()`][mesqual.datasets.DatasetLinkCollection.get_dataset_by_type]
    """

    def __init__(
        self,
        datasets: list[DatasetType],
        name: str = None,
        parent_dataset: Dataset = None,
        attributes: dict = None,
        cache: Cache = None,
        config: DatasetConfigType = None,
    ):
        """Create a link collection.

        The arguments are the same as for `DatasetCollection`.

        Raises:
            AmbiguousOverrideError: Two children accept the same flag.
            ValueError: Two children have the same name or different specs
                with the same name, or a `model_flag` names no model spec.
        """
        super().__init__(
            datasets=datasets,
            name=name,
            parent_dataset=parent_dataset,
            attributes=attributes,
            cache=cache,
            config=config,
        )
        self._raise_if_flags_overlap()
        if self.datasets:
            self.flag_index.check_references()

    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        child = self._children_to_fetch(flag)[0]
        return child.fetch(flag, effective_config, **self._kwargs_for_child(child, flag))

    def _raise_if_flags_overlap(self) -> None:
        counts = Counter(flag for ds in self.datasets for flag in ds.accepted_flags)
        duplicates = sorted(str(flag) for flag, count in counts.items() if count > 1)
        if duplicates:
            raise AmbiguousOverrideError(
                f'Link collection {self.name}: several children accept the flags {duplicates}. '
                f'Each flag must route to one child. Use a merge or concat collection to combine '
                f'datasets with the same flags.'
            )

    def get_dataset_by_type(self, ds_type: type[Dataset]) -> DatasetType:
        """Returns instance of child dataset that matches the ds_type."""
        for ds in self.datasets:
            if isinstance(ds, ds_type):
                return ds
        raise KeyError(f'No Dataset of type {ds_type.__name__} found in {self.name}.')


class DatasetMergeCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
):
    """Merges the results of all children that have a flag into one result.

    Use it for fragmented data of one scenario, for example simulation runs
    per month, or a model table from two sources.

    The results combine pairwise (`combine_dfs`): disjoint columns with
    overlapping index concat on columns; shared columns with disjoint index
    concat on rows; else `combine_first`. Where two children overlap,
    `precedence` decides which values stay. With a name list as
    `precedence`, the columns of disjoint results also follow the order of
    the list, and so do their rows if `auto_sort_index` is `False`.

    Examples:
        Two periods of one scenario:

        >>> from mesqual.testing import DummyDataset, DummyScenario
        >>> first, second = DummyScenario(days=4).split('2024-01-03')
        >>> merged = DatasetMergeCollection([DummyDataset(first), DummyDataset(second)], name='base')
        >>> len(merged.fetch('Node.Results.price'))  # 4 days of hours
        96

        A rerun of the first day replaces the values of the base run there:

        >>> base = DummyDataset(DummyScenario(days=4))
        >>> rerun = DummyDataset(DummyScenario(name='rerun', days=1, demand_factor=1.2))
        >>> patched = DatasetMergeCollection([base, rerun], precedence=['rerun', 'base'])
        >>> patched.fetch('Node.Results.demand_total').iloc[0].equals(rerun.fetch('Node.Results.demand_total').iloc[0])
        True
    """

    def __init__(
        self,
        datasets: list[DatasetType],
        name: str = None,
        parent_dataset: Dataset = None,
        attributes: dict = None,
        cache: Cache = None,
        config: DatasetConfigType = None,
        precedence: Literal['first', 'last'] | Sequence[str] = 'first',
    ):
        """Create a merge collection.

        Args:
            datasets: The child datasets.
            name: Name of the collection.
            parent_dataset: Optional parent dataset.
            attributes: Attributes of the collection.
            cache: Optional cache of fetch results.
            config: Instance config.
            precedence: Which child wins where two children overlap:
                `'first'` (default) the earlier child, `'last'` the later
                child, or a list (or other iterable) of all child names,
                highest precedence first.

        Raises:
            ValueError: `precedence` is not `'first'`, `'last'` or a list
                with each child name once.
        """
        super().__init__(
            datasets=datasets,
            name=name,
            parent_dataset=parent_dataset,
            attributes=attributes,
            cache=cache,
            config=config,
        )
        is_name_iterable = isinstance(precedence, Iterable) and not isinstance(precedence, str)
        self.precedence = list(precedence) if is_name_iterable else precedence
        self._check_precedence([ds.name for ds in self.datasets])

    def add_dataset(self, dataset: DatasetType):
        """Add a child dataset. A child with the same name replaces the existing one and logs a warning.

        With a name list as `precedence`, first set `precedence` to a list
        that also names the new child, then add the child.

        Raises:
            TypeError: The dataset is not of `get_child_dataset_type()`.
            ValueError: `precedence` is a name list that does not name each
                child once with the new child; a name is a flag parameter of
                one dataset and a dataset parameter (`fetch_params`) of
                another in the collection; or the new child and another child
                have different specs with the same name.
        """
        self._check_precedence({ds.name for ds in self.datasets} | {dataset.name})
        super().add_dataset(dataset)

    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        children = self._children_by_precedence(self._children_to_fetch(flag))
        return combine_dfs(
            [ds.fetch(flag, effective_config, **self._kwargs_for_child(ds, flag)) for ds in children],
            precedence='last' if self.precedence == 'last' else 'first',
        )

    def _check_precedence(self, child_names: Iterable[str]) -> None:
        """Raise `ValueError` if `precedence` is not `'first'`, `'last'` or a list with each of `child_names` once."""
        if self.precedence in ('first', 'last'):
            return
        child_names = sorted(child_names)
        if not isinstance(self.precedence, list) or sorted(self.precedence) != child_names:
            raise ValueError(
                f"Merge collection {self.name}: precedence must be 'first', 'last' or a list of all child "
                f'names, got {self.precedence!r}. Children: {child_names}.'
            )

    def _children_by_precedence(self, children: list[DatasetType]) -> list[DatasetType]:
        """`children` in the order for `combine_dfs`: as given for `'first'` and `'last'`, else as in the name list.

        Raises:
            ValueError: `precedence` is not `'first'`, `'last'` or a list
                with each child name once.
        """
        self._check_precedence([ds.name for ds in self.datasets])
        if self.precedence in ('first', 'last'):
            return children
        rank = {name: position for position, name in enumerate(self.precedence)}
        return sorted(children, key=lambda ds: rank[ds.name])


class DatasetConcatCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
):
    """
    Concatenates the results of its child datasets, with a level that names the child.

    It fetches the same flag from each child dataset and concatenates the
    results into one DataFrame or Series. The scenarios and the comparisons
    of a study are concat collections.

    MultiIndex Structure:
        The result has a new level named `concat_level_name` (default
        `dataset`) with the child names, at the top of the axis
        `concat_axis`: `1` (default) concats the child results as columns,
        `0` stacks them as rows. The axis is fixed per collection. Each
        nested concat collection uses its own axis.
        `concat_top=False` puts it last. The join is outer, so children with
        different horizons give `NaN` outside their horizon.

        The level names of the children must match on each axis. A child
        whose result lacks levels of the longest child result, with the
        other levels in the same order, gets these levels with the value
        `''`. So a comparison with `comparison_type=BOTH` concats with
        scenarios. If the child result and the longest child result have
        unique level names, a child result with only level names of the
        longest child result, in another order, raises `ValueError` that
        names both children and both orders. Other differences raise
        `NotImplementedError`.

        With `auto_sort_index`, a concat on the columns sorts the rows. A
        concat on the rows keeps the order of the children; each child
        result is sorted already.

    Examples:
        >>> prices = study.scen.fetch('Node.Results.price')
        >>> list(prices.columns.names)
        ['dataset', 'Node']
        >>> base_prices = prices['base']

        Stack the scenarios as rows:

        >>> stacked = DatasetConcatCollection(study.scen.datasets, name='stacked', concat_axis=0)
        >>> list(stacked.fetch('Node.Results.price').index.names)
        ['dataset', 'time']

    Child parameters:
        It passes each child the dataset parameters that the child accepts
        and that the call or `param_defaults` gave, for example
        `comparison_type` of comparisons.
    """

    DEFAULT_CONCAT_LEVEL_NAME = 'dataset'
    DEFAULT_ATT_LEVEL_NAME = 'attribute'

    def __init__(
        self,
        datasets: list[DatasetType],
        name: str = None,
        parent_dataset: Dataset = None,
        attributes: dict = None,
        cache: Cache = None,
        config: DatasetConfigType = None,
        concat_axis: int = 1,
        concat_top: bool = True,
        concat_level_name: str = None,
    ):
        """Create a concat collection.

        Args:
            datasets: The child datasets. Their names must be unique.
            name: Name of the collection.
            parent_dataset: Optional parent dataset.
            attributes: Attributes of the collection.
            cache: Optional cache of fetch results.
            config: Instance config.
            concat_axis: Axis of the concat: `1` (default) concats the
                child results as columns, `0` stacks them as rows. A
                Python or numpy integer; `bool` and `float` values raise.
            concat_top: `True` (default) puts the new level at the top of
                the concat axis, `False` at the bottom. Stored as `bool`.
            concat_level_name: Name of the new level. Default `dataset`.

        Raises:
            ValueError: `concat_axis` is not `0` or `1`, two children have
                the same name, or two children have different specs with
                the same name.
        """
        is_integer = isinstance(concat_axis, (int, np.integer)) and not isinstance(concat_axis, (bool, np.bool_))
        if not is_integer or concat_axis not in (0, 1):
            raise ValueError(f'concat_axis must be 0 (rows) or 1 (columns), got {concat_axis!r}.')
        super().__init__(
            datasets=datasets,
            name=name,
            parent_dataset=parent_dataset,
            attributes=attributes,
            cache=cache,
            config=config,
        )
        self._concat_axis = int(concat_axis)
        self._concat_top = bool(concat_top)
        self._concat_level_name = concat_level_name or self.DEFAULT_CONCAT_LEVEL_NAME

    @property
    def concat_axis(self) -> int:
        """Axis of the concat, fixed at construction: `1` columns, `0` rows."""
        return self._concat_axis

    @property
    def concat_top(self) -> bool:
        """`True` if the new level is at the top of the concat axis, `False` if at the bottom. Fixed at construction."""
        return self._concat_top

    @property
    def concat_level_name(self) -> str:
        """Name of the new level with the child names. Fixed at construction."""
        return self._concat_level_name

    def transform_schema(self, schema: FrameSchema) -> FrameSchema:
        """Add `Level(concat_level_name, 'dataset')` to the axis of the concat.

        The axis is `concat_axis` of the collection. The level is at the
        top, or at the bottom with `concat_top=False`.
        """
        level = Level(self.concat_level_name, 'dataset')
        axis = 'index' if self.concat_axis == 0 else 'columns'
        levels = getattr(schema, axis)
        levels = (level, *levels) if self.concat_top else (*levels, level)
        return dataclasses.replace(schema, **{axis: levels})

    def _sort_index(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Sort the rows, unless the collection concats on rows (`concat_axis=0`): the concat level keeps the order of the children."""
        if self.concat_axis == 0:
            return data
        return super()._sort_index(data)

    def version_identity(self, flag: FlagType) -> tuple:
        """As `DatasetCollection.version_identity`, with `('concat', concat_axis, concat_top, concat_level_name)` first.

        So a collection with the same name and another layout of its result
        does not reuse a cache entry.
        """
        layout = ('concat', self.concat_axis, self.concat_top, self.concat_level_name)
        return (layout, *super().version_identity(flag))

    def get_attributes_concat_df(self) -> pd.DataFrame:
        if all(isinstance(ds, DatasetConcatCollection) for ds in self.datasets):
            use_att_df_instead_of_series = True
        else:
            use_att_df_instead_of_series = False

        atts_per_dataset = dict()
        for ds in self.datasets:
            atts = ds.get_attributes_concat_df().T if use_att_df_instead_of_series else ds.get_attributes_series()
            atts_per_dataset[ds.name] = atts

        return (
            pd.concat(atts_per_dataset, axis=1, names=[self.concat_level_name])
            .rename_axis(self.DEFAULT_ATT_LEVEL_NAME)
            .T
        )

    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        dfs = {
            ds.name: ds.fetch(flag, effective_config, **self._kwargs_for_child(ds, flag))
            for ds in self._children_to_fetch(flag)
        }
        dfs = self._align_levels(dfs)
        df = pd.concat(dfs, join='outer', axis=self.concat_axis, names=[self.concat_level_name])
        return df if self.concat_top else move_top_level_last(df, self.concat_axis)

    @classmethod
    def _align_levels(cls, dfs: dict[str, pd.Series | pd.DataFrame]) -> dict[str, pd.Series | pd.DataFrame]:
        """Give all results the same level names on each axis.

        A result whose level names are the names of the longest result
        without some levels, in the same order, gets the missing levels with
        the value `''`.

        Raises:
            ValueError: A result has only level names of the longest result,
                but in another order. The result and the longest result have
                unique level names. The message names both children and both
                orders.
            NotImplementedError: The results differ in kind, or their level
                names differ in another way.
        """
        first = next(iter(dfs.values()))
        if not all(len(df.axes) == len(first.axes) for df in dfs.values()):
            raise NotImplementedError('Axes lengths do not match between dfs.')
        for axis in range(len(first.axes)):
            names = {key: list(df.axes[axis].names) for key, df in dfs.items()}
            target_key = max(names, key=lambda key: len(names[key]))
            target = names[target_key]
            target_is_unique = len(set(target)) == len(target)
            for key, own in names.items():
                if own == target:
                    continue
                own_is_unique = len(set(own)) == len(own)
                if not (target_is_unique and own_is_unique and set(own) <= set(target)):
                    raise NotImplementedError(f'Axes names do not match between dfs: {own} and {target}.')
                if not cls._is_subsequence(own, target):
                    raise ValueError(
                        f'Child {key!r} has the levels {own} on axis {axis} in another order than child '
                        f'{target_key!r} {target}. Give the child results the same level order.'
                    )
                dfs[key] = dfs[key].set_axis(cls._padded_axis(dfs[key].axes[axis], target), axis=axis)
        return dfs

    @staticmethod
    def _is_subsequence(names: list, target: list) -> bool:
        remaining = iter(target)
        return all(name in remaining for name in names)

    @staticmethod
    def _padded_axis(ax: pd.Index, target: list) -> pd.MultiIndex:
        """`ax` with the levels of `target` that it lacks, each with the value `''`."""
        own = list(ax.names)
        empty = np.full(len(ax), '', dtype=object)
        arrays = [ax.get_level_values(own.index(name)) if name in own else empty for name in target]
        return pd.MultiIndex.from_arrays(arrays, names=target)


class DatasetSumCollection(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
):
    """Sums the time series flags of its children and merges their model flags.

    Time series of numeric data add element-wise. An object or row that one
    child lacks gets the `fill_value` of the spec before the addition; a
    spec without `fill_value` gives `NaN` there. Model flags merge as in
    `DatasetMergeCollection`, the first child wins.
    """

    def _fetch(self, flag: FlagType, effective_config: DatasetConfigType, **kwargs) -> pd.Series | pd.DataFrame:
        children = self._children_to_fetch(flag)
        item_type = self.flag_index.get_item_type(flag)
        if item_type not in (ItemTypeEnum.Model, ItemTypeEnum.TimeSeries):
            raise NotImplementedError(
                f'Sum collection {self.name}: flag {flag} has item type {item_type.name}. '
                f'A sum collection sums time series flags and merges model flags only.'
            )
        data = [ds.fetch(flag, effective_config, **self._kwargs_for_child(ds, flag)) for ds in children]
        if item_type is ItemTypeEnum.Model:
            return combine_dfs(data, precedence='first')
        not_numeric = [ds.name for ds, d in zip(children, data) if not pd_is_numeric(d)]
        if not_numeric:
            raise TypeError(
                f'Sum collection {self.name}: flag {flag} has non-numeric data in {not_numeric}. '
                f'A sum collection adds numeric time series only.'
            )
        fill_value = self.flag_index.get_fill_value(flag)
        return reduce(lambda a, b: a.add(b, fill_value=fill_value), data)
