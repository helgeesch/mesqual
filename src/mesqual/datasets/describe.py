"""Describe methods: flags, parameters, layers and datasets, without time series data."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import pandas as pd

from mesqual.enums import ItemTypeEnum, QuantityTypeEnum
from mesqual.flag.dimension import Dimension
from mesqual.flag.layer_spec import LayerFlagSpec
from mesqual.flag.flag_index import FlagIndex
from mesqual.flag.flag_spec import FlagSpec, ModelFlagSpec
from mesqual.flag.layers import AreaLayer, BorderLayer
from mesqual.flag.schema import FrameSchema, Level
from mesqual.units import Units

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset
    from mesqual.datasets.fetch_context import FetchRecord

RESOLVED_PER_DATASET = 'resolved per dataset'
"""Values of a dimension whose values come from a function of the dataset, shown on a dataset class."""


@dataclass(frozen=True)
class Availability:
    """Whether a dataset can answer a flag.

    Attributes:
        available: `True` or `False`, or `None` for unknown.
        reason: Where the answer comes from.
        datasets: For a collection: names of the children on which the flag
            is available or unknown. `None` for other datasets.
    """

    available: bool | None
    reason: str
    datasets: tuple[str, ...] | None = None

    @property
    def label(self) -> str:
        """`'available'`, `'unavailable'` or `'unknown'`."""
        return {True: 'available', False: 'unavailable', None: 'unknown'}[self.available]

    @classmethod
    def union(cls, by_child: Mapping[str, Availability]) -> Availability:
        """Combine the availability of the children of a collection.

        Available if one child has the flag available, else unknown if one
        child does not know, else unavailable.
        """
        states = {availability.available for availability in by_child.values()}
        available = True if True in states else (None if None in states else False)
        groups = {
            label: [name for name, availability in by_child.items() if availability.label == label]
            for label in ('available', 'unknown', 'unavailable')
        }
        reason = '; '.join(f'{label} in {", ".join(names)}' for label, names in groups.items() if names)
        datasets = tuple(name for name, availability in by_child.items() if availability.available is not False)
        return cls(available, reason or 'no children', datasets)

    @classmethod
    def intersection(cls, by_child: Mapping[str, Availability]) -> Availability:
        """Combine the availability of children that are all needed, for example of a comparison.

        Unavailable if one child has the flag unavailable, else unknown if
        one child does not know, else available.
        """
        available = cls._weakest_state(by_child.values()) if by_child else False
        reason = '; '.join(f'{availability.label} in {name}' for name, availability in by_child.items())
        datasets = () if available is False else tuple(by_child)
        return cls(available, reason or 'no children', datasets)

    @classmethod
    def of_inputs(cls, by_flag: Mapping[str, Availability]) -> Availability:
        """Combine the availability of the flags that a flag needs.

        Unavailable if one input is unavailable, else unknown if one input
        is unknown, else available. The reason names the unavailable or the
        unknown inputs with their reasons.

        Args:
            by_flag: The availability of each input flag.

        Examples:
            >>> Availability.of_inputs({
            ...     'Node.Results.price': Availability(True, 'prices in the source'),
            ...     'Node.Results.demand': Availability(False, 'no demand table'),
            ... }).reason
            'unavailable input Node.Results.demand (no demand table)'
        """
        available = cls._weakest_state(by_flag.values())
        if available is True:
            return cls(True, f'inputs available: {", ".join(by_flag)}' if by_flag else 'no inputs')
        label = 'unavailable' if available is False else 'unknown'
        reason = '; '.join(
            f'{label} input {flag} ({availability.reason})'
            for flag, availability in by_flag.items()
            if availability.available is available
        )
        return cls(available, reason)

    @staticmethod
    def _weakest_state(availabilities: Iterable[Availability]) -> bool | None:
        """`False` if one is unavailable, else `None` if one is unknown, else `True`."""
        states = {availability.available for availability in availabilities}
        return False if False in states else (None if None in states else True)


@dataclass(frozen=True)
class ParamInfo:
    """One parameter of a flag: a flag parameter of the spec or a dataset parameter of the dataset class.

    Attributes:
        name: Keyword argument of `fetch`.
        owner: `'spec'` for a flag parameter, `'dataset'` for a dataset parameter (`fetch_params`).
        description: What the parameter selects.
        values: Allowed values with their descriptions, `None` for an open
            parameter, or `'resolved per dataset'` on a dataset class.
        default: Effective default. `None` for a required parameter.
        default_source: `'spec'`, `'dataset'` (the default of the dimension),
            `'config'` (`param_defaults`), `'required'` (no default), or
            `'per dataset'` on a collection whose children resolve
            different defaults (`default` is then `None`).
    """

    name: str
    owner: str
    description: str
    values: Mapping[Any, str] | str | None
    default: Any
    default_source: str

    def as_record(self) -> dict:
        """Return the parameter as a plain dict, ready for JSON.

        `values` is a list of `{'value': ..., 'description': ...}`, so the
        default is equal to the `value` of one entry.
        """
        record = _Plain.of(self.__dict__)
        if isinstance(self.values, Mapping):
            record['values'] = [
                {'value': _Plain.of(value), 'description': description} for value, description in self.values.items()
            ]
        return record

    def __str__(self) -> str:
        return f'{self.name}={self.default!r} ({self.default_source})'


@dataclass(frozen=True)
class FlagInfo:
    """Everything mesqual knows about one flag of a dataset, without its data.

    Attributes:
        flag: The flag.
        spec: The flag spec.
        base: The base spec of a layer spec, else `None`.
        aggregation: The aggregation of a layer spec, else `None`.
        item_type: Item type of the flag.
        variant: The variant values of the flag.
        description: Spec description and one line per variant value.
        unit: Unit of the values, `None` for a model flag.
        unit_source: `'spec'`, or `'base'` for a layer spec.
        quantity_type: Quantity type of the values.
        quantity_type_source: `'spec'` (an ambiguous unit), `'unit'`, or `'base'` for a layer spec.
        schema: Index and column levels of the result.
        params: The flag parameters and the dataset parameters, with their effective defaults.
        fill_value: Value for objects or rows that one side lacks, `None` for no fill.
        model_flag: The linked model flag, or `None`.
        available: `True`, `False` or `None` for unknown.
        availability_reason: Where the availability comes from.
        chain: For a platform dataset: the interpreter chain of the flag,
            from the original interpreter to the one that answers.
        dependencies: The recorded dependencies of the last fetch of the
            flag without arguments. Empty before the first fetch.
    """

    flag: str
    spec: FlagSpec
    base: FlagSpec | None
    aggregation: Any
    item_type: ItemTypeEnum
    variant: dict[str, Any]
    description: str
    unit: Units.Unit | None
    unit_source: str
    quantity_type: QuantityTypeEnum
    quantity_type_source: str
    schema: FrameSchema | None
    params: tuple[ParamInfo, ...]
    fill_value: float | None
    model_flag: str | None
    available: bool | None
    availability_reason: str
    chain: tuple[str, ...] = ()
    dependencies: tuple[FetchRecord, ...] = field(default=(), repr=False)

    def as_record(self) -> dict:
        """Return the info as a plain dict, ready for JSON.

        Spec, base and aggregation appear by name, the schema as
        `{'index': [level, ...], 'columns': [level, ...], 'dtype': ...}`
        with one dict per level (`name`, `role`).
        """
        record = {k: v for k, v in self.__dict__.items() if k not in ('spec', 'base', 'aggregation', 'dependencies')}
        record['spec'] = self.spec.name
        record['base'] = None if self.base is None else self.base.name
        record['aggregation'] = None if self.aggregation is None else type(self.aggregation).__name__
        record['params'] = [p.as_record() for p in self.params]
        record['dependencies'] = [f'{d.dataset_name}: {d.flag}' for d in self.dependencies]
        return _Plain.of(record)


class _Plain:
    """Conversion of describe values to plain Python values, ready for JSON."""

    @classmethod
    def of(cls, value: Any) -> Any:
        """Return `value` with enums as names, mappings and schemas as dicts, other objects as strings."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, ParamInfo):
            return value.as_record()
        if isinstance(value, (FrameSchema, Level)):
            return {f.name: cls.of(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, Mapping):
            return {str(cls.of(k)): cls.of(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls.of(v) for v in value]
        return str(value)

    @classmethod
    def text(cls, value: Any) -> Any:
        """Return a short text for a table cell: lists joined with commas, mappings as `key: values`."""
        if isinstance(value, Mapping):
            return '; '.join(f'{k}: {cls.text(v)}' for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return ', '.join(str(v) for v in value)
        return value


class FlagDescriber:
    """Builds the rows of the describe methods of a dataset class or instance.

    It reads specs, the availability of the dataset and model flags (for flag
    families), never time series data.

    Args:
        owner: A dataset instance, or a dataset class for the `describe_class_*` methods.
    """

    LAYER_COLUMNS = ('index', 'refers_to', 'mapping_columns', 'flags', 'description')

    def __init__(self, owner: Dataset | type[Dataset]):
        self._owner = owner
        self._is_class = isinstance(owner, type)

    @staticmethod
    def name_contains(name: Any, x: str, match_case: bool = False) -> bool:
        """`True` if `x` is a substring of `name`, case-insensitive unless `match_case`."""
        if match_case:
            return x in str(name)
        return x.lower() in str(name).lower()

    def class_spec_rows(self) -> list[dict]:
        """Rows of `describe_class_flags`: the specs that the class declares.

        Raises:
            TypeError: The class is a collection class other than a platform dataset class.
        """
        self._check_own_specs('describe_class_flags')
        return self.spec_rows(self._owner._declared_flag_specs())

    def _check_own_specs(self, method: str) -> None:
        if not self._owner._describes_class_flags():
            raise TypeError(
                f'{self._owner.__name__} is a collection class and has no flags of its own: {method} '
                f'does not apply. Use describe_flags() on the instance.'
            )

    def spec_rows(self, specs: Iterable[FlagSpec]) -> list[dict]:
        """One row per spec, as declared on the class."""
        rows = []
        for spec in specs:
            layers = [layer.name for layer in self._index().get_layers(spec)]
            rows.append(
                {**self._spec_fields(spec, layers), 'variants': self._declared_variants(spec), 'layers': layers}
            )
        return rows

    def flag_rows(self) -> list[dict]:
        """One row per spec of the dataset, with resolved variants, layers, parameters and availability."""
        rows = []
        for name, by_flag in self._owner._flag_availability_by_spec().items():
            spec = self._owner.flag_index.get_spec_by_name(name)
            rows.append(self._flag_row(spec, by_flag))
        return rows

    def flag_info(self, flag: Any) -> FlagInfo:
        """Return the `FlagInfo` of one flag of the dataset."""
        dataset = self._owner
        spec = dataset.flag_index.get_spec(flag)
        availability = dataset.availability(flag)
        base = getattr(spec, 'base', None)
        quantity_type, quantity_type_source = self._quantity_type(spec)
        return FlagInfo(
            flag=str(flag),
            spec=spec,
            base=base,
            aggregation=getattr(spec, 'aggregation', None),
            item_type=spec.item_type,
            variant=spec.variant_of(flag).as_dict(),
            description=self._description(spec, flag),
            unit=getattr(spec, 'unit', None),
            unit_source='base' if base is not None else 'spec',
            quantity_type=quantity_type,
            quantity_type_source=quantity_type_source,
            schema=dataset._result_schema(flag),
            params=self._params(spec, flag),
            fill_value=dataset.flag_index.get_fill_value(flag),
            model_flag=dataset.flag_index.get_linked_model_flag(flag),
            available=availability.available,
            availability_reason=availability.reason,
            chain=dataset._flag_chain_names(flag),
            dependencies=self._dependencies(flag),
        )

    def _description(self, spec: FlagSpec, flag: Any) -> str:
        """The combined description, or the spec description for a family value that the dataset does not have."""
        try:
            return spec.flag_description(flag, self._owner._values_dataset(flag))
        except (KeyError, ValueError):
            return spec.description

    def area_layer_rows(self) -> list[dict]:
        """Rows of `describe_class_area_layers`: one row per area layer registered on the class."""
        self._check_own_specs('describe_class_area_layers')
        index = self._index()
        return [
            self._layer_row(layer, f'{layer.base_model} -> {layer.layer_model}', [index.get_mapping_column(layer)])
            for layer in index.layers.area_layers
        ]

    def border_layer_rows(self) -> list[dict]:
        """Rows of `describe_class_border_layers`: one row per border layer registered on the class."""
        self._check_own_specs('describe_class_border_layers')
        index = self._index()
        return [
            self._layer_row(layer, f'{layer.area_layer} via {layer.edge_model}', list(index.get_edge_ends(layer)))
            for layer in index.layers.border_layers
        ]

    @staticmethod
    def filter_rows(rows: list[dict], contains: str | None, layer: str | None, item_type: Any) -> list[dict]:
        """Keep the rows that match all given filters.

        Args:
            rows: Rows with the keys `name`, `flags_list`, `layers` and `item_type`.
            contains: Substring of the name or of one flag, case-insensitive.
            layer: Name of a layer in `layers`.
            item_type: An `ItemTypeEnum` or its value.
        """
        if contains is not None:
            rows = [
                r for r in rows if any(FlagDescriber.name_contains(n, contains) for n in (r['name'], *r['flags_list']))
            ]
        if layer is not None:
            rows = [r for r in rows if layer in r['layers']]
        if item_type is not None:
            wanted = item_type.value if isinstance(item_type, ItemTypeEnum) else str(item_type)
            rows = [r for r in rows if r['item_type'] == wanted]
        return rows

    @staticmethod
    def output(
        rows: list[dict],
        as_records: bool,
        index: str = 'name',
        hidden: tuple[str, ...] = (),
        columns: tuple[str, ...] = (),
    ) -> Any:
        """Return the rows as a `DataFrame` for people, or as plain dicts with `as_records`.

        Args:
            rows: The rows.
            as_records: Return plain dicts.
            index: Column that becomes the index of the `DataFrame`.
            hidden: Keys to drop.
            columns: Columns of an empty `DataFrame`.
        """
        rows = [{k: v for k, v in row.items() if k not in hidden} for row in rows]
        if as_records:
            return [_Plain.of(row) for row in rows]
        frame = pd.DataFrame.from_records(
            [{k: _Plain.text(v) for k, v in row.items()} for row in rows],
            columns=list(rows[0]) if rows else [index, *columns],
        )
        return frame.set_index(index)

    def _spec_fields(self, spec: FlagSpec, layers: list[str]) -> dict:
        quantity_type, _ = self._quantity_type(spec)
        unit = getattr(spec, 'unit', None)
        return {
            'name': spec.name,
            'item_type': spec.item_type.value,
            'object': self._object_of(spec, layers),
            'unit': None if unit is None else str(unit),
            'quantity_type': quantity_type.name,
            'params': list(self._params(spec)),
            'description': spec.description,
            'flags_list': sorted(spec.static_flags()),
        }

    def _flag_row(self, spec: FlagSpec, by_flag: Mapping[str, Availability]) -> dict:
        reachable = [f for f, a in by_flag.items() if a.available is not False]
        layers = self._layers_of(spec, [spec.variant_of(f).as_dict() for f in reachable])
        row = self._spec_fields(spec, layers)
        row['variants'] = self._resolved_variants(spec, reachable)
        row['layers'] = layers
        if reachable:
            row['params'] = list(self._params(spec, reachable[0]))
        row['flags'] = len(reachable)
        row['flags_list'] = sorted(by_flag)
        row['availability'] = self._row_availability(by_flag.values())
        row['datasets'] = self._row_datasets(by_flag.values())
        return row

    @staticmethod
    def _row_availability(states: Iterable[Availability]) -> str:
        labels = {a.label for a in states}
        if not labels or labels == {'unavailable'}:
            return 'unavailable'
        if 'unavailable' in labels:
            return 'partial'
        return 'unknown' if 'unknown' in labels else 'available'

    def _row_datasets(self, states: Iterable[Availability]) -> list[str] | None:
        """Children that have the flag, for a collection whose children do not all have it."""
        children = self._owner._describe_children()
        names = {n for a in states for n in (a.datasets or ())}
        if not children or names == {ds.name for ds in children}:
            return None
        return [ds.name for ds in children if ds.name in names]

    def _object_of(self, spec: FlagSpec, layers: list[str]) -> str | list[str] | None:
        """The model flag of the spec: the objects in the rows or columns of its result.

        A model spec gives itself. A layer spec without `model_flag` gives
        the layer model flags of `layers`, in the same order: the area model
        of an area layer, the border model of a border layer. A border layer
        without a border model spec gives none.
        """
        if isinstance(spec, ModelFlagSpec):
            return spec.name
        if not isinstance(spec, LayerFlagSpec) or spec.model_flag is not None:
            return getattr(spec, 'model_flag', None)
        index = self._index()
        model_flags = (index.get_layer_model_flag(index.layers.get(name)) for name in layers)
        return [flag for flag in dict.fromkeys(model_flags) if flag is not None]

    def _index(self) -> FlagIndex:
        """The flag index of the instance, or the class flag index of the class."""
        return self._owner._class_flag_index() if self._is_class else self._owner.flag_index

    @staticmethod
    def _quantity_type(spec: FlagSpec) -> tuple[QuantityTypeEnum, str]:
        source = 'base' if getattr(spec, 'base', None) is not None else None
        quantity_type = getattr(spec, 'quantity_type', None)
        if quantity_type is not None:
            return quantity_type, source or 'spec'
        unit = getattr(spec, 'unit', None)
        if unit is None:
            return QuantityTypeEnum.UNKNOWN, source or 'unit'
        return Units.get_quantity_type_enum(unit), source or 'unit'

    def _declared_variants(self, spec: FlagSpec) -> dict[str, Any]:
        return {d.name: RESOLVED_PER_DATASET if d.is_resolved_per_dataset else list(d.values) for d in spec.variants}

    @staticmethod
    def _resolved_variants(spec: FlagSpec, flags: list[str]) -> dict[str, list]:
        values = {d.name: [] for d in spec.variants}
        for flag in flags:
            for name, value in spec.variant_of(flag).as_dict().items():
                if value not in values[name]:
                    values[name].append(value)
        return values

    def _params(self, spec: FlagSpec, flag: Any = None) -> tuple[ParamInfo, ...]:
        """Flag parameters of the spec and dataset parameters of the dataset class, with effective defaults.

        On an instance and with a flag, each default is the one that a fetch
        of the flag resolves (`_resolved_default`).
        """
        owner = self._owner
        dataset_params = owner.fetch_params if self._is_class else tuple(owner._accepted_dataset_params.values())
        values_dataset = None if self._is_class or flag is None else owner._values_dataset(flag)
        return tuple(
            self._param(dimension, kind, self._default_of(dimension, kind, flag), values_dataset)
            for kind, dimensions in (('spec', spec.params), ('dataset', dataset_params))
            for dimension in dimensions
        )

    def _default_of(self, dimension: Dimension, owner: str, flag: Any) -> tuple[Any, str]:
        if self._is_class:
            return self._own_default(dimension, owner, {})
        if flag is None:
            return self._own_default(dimension, owner, self._owner.instance_config.param_defaults or {})
        return self._resolved_default(self._owner, dimension, owner, flag, None)

    @classmethod
    def _resolved_default(
        cls,
        dataset: Dataset,
        dimension: Dimension,
        owner: str,
        flag: Any,
        outer_config: Any,
    ) -> tuple[Any, str]:
        """The default and its source that a fetch of `flag` resolves for one parameter.

        A dataset that resolves the parameter itself uses its effective
        config under `outer_config`, as `fetch` does. A collection asks each
        child that fetches the flag and accepts the parameter, with the
        effective config of the collection as outer config. If the children
        give different results, the source is `'per dataset'`.
        """
        effective = dataset._prepare_config(outer_config)
        if owner == 'spec':
            resolves_here = dataset._resolves_flag_params
        else:
            resolves_here = dimension.name in {d.name for d in dataset.fetch_params}
        children = (
            []
            if resolves_here
            else [
                child
                for child in getattr(dataset, 'datasets', ())
                if dataset._fetches_from(child, flag)
                and (owner == 'spec' or dimension.name in child._accepted_dataset_params)
            ]
        )
        if not children:
            return cls._own_default(dimension, owner, effective.param_defaults or {})
        results = []
        for child in children:
            result = cls._resolved_default(child, dimension, owner, flag, effective)
            if result not in results:
                results.append(result)
        return results[0] if len(results) == 1 else (None, 'per dataset')

    @staticmethod
    def _own_default(dimension: Dimension, owner: str, param_defaults: Mapping[str, Any]) -> tuple[Any, str]:
        if dimension.name in param_defaults:
            return param_defaults[dimension.name], 'config'
        if dimension.has_default:
            return dimension.default, owner
        return None, 'required'

    @staticmethod
    def _param(dimension: Dimension, owner: str, default: tuple[Any, str], values_dataset: Any) -> ParamInfo:
        if dimension.is_resolved_per_dataset:
            values = RESOLVED_PER_DATASET if values_dataset is None else dimension.values_for(values_dataset)
        else:
            values = dimension.values
        return ParamInfo(dimension.name, owner, dimension.description, values, *default)

    @staticmethod
    def _layers_of(spec: FlagSpec, variants: Iterable[Mapping[str, Any]]) -> list[str]:
        placeholder = getattr(spec, 'layer_placeholder', None)
        layers = []
        for variant in variants if placeholder else ():
            if variant[placeholder] not in layers:
                layers.append(variant[placeholder])
        return layers

    def _layer_row(self, layer: AreaLayer | BorderLayer, refers_to: str, columns: list[str]) -> dict:
        index = self._index()
        return {
            'name': layer.name,
            'index': layer.index,
            'refers_to': refers_to,
            'mapping_columns': columns,
            'flags': sum(layer in index.get_layers(spec) for spec in index.specs),
            'description': layer.description,
        }

    def _dependencies(self, flag: Any) -> tuple:
        try:
            record = self._owner.get_fetch_record(flag)
        except ValueError:
            return ()
        return () if record is None else tuple(record.dependencies)


class DatasetTreeDescriber:
    """Builds the rows of `describe_datasets`: one row per dataset of the tree.

    Args:
        root: The dataset at the top of the tree.
    """

    def __init__(self, root: Dataset):
        self._root = root

    def rows(self, max_depth: int | None = None) -> list[dict]:
        """One row per dataset, depth first, down to `max_depth` levels below the root (`None`: all).

        The path joins the names from the root with `/`.
        """
        return list(self._rows_of(self._root, None, max_depth))

    def _rows_of(self, dataset: Dataset, parent_path: str | None, max_depth: int | None) -> Iterable[dict]:
        from mesqual.datasets.dataset_config import DatasetConfig

        path = dataset.name if parent_path is None else f'{parent_path}/{dataset.name}'
        children = dataset._describe_children()
        yield {
            'path': path,
            'name': dataset.name,
            'class': type(dataset).__qualname__,
            'description': dataset.description,
            'attributes': dict(dataset.attributes),
            'config': dict(DatasetConfig._explicit_items(dataset._config)),
            'children': [child.name for child in children],
        }
        if max_depth == 0:
            return
        for child in children:
            yield from self._rows_of(child, path, None if max_depth is None else max_depth - 1)
