"""Lookup from a flag name to the flag spec that declares it."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any, Iterable

from mesqual.enums import ItemTypeEnum, QuantityTypeEnum, TopologyTypeEnum, VisualizationTypeEnum
from mesqual.flag.layer_spec import AreaFlagSpec, BorderFlagSpec, LayerFlagSpec, LayerVariantValues
from mesqual.flag.exceptions import AmbiguousOverrideError, UnknownFlagError
from mesqual.flag.flag_spec import FlagSpec, ModelFlagSpec, TimeSeriesFlagSpec
from mesqual.flag.layers import AreaLayer, BorderLayer, LayerSet
from mesqual.flag.schema import FrameSchema, Level
from mesqual.units import Units

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset


class FlagIndex:
    """Lookup from a flag name to its flag spec, and the getters that read the spec.

    The flag index has no implicit rules: every answer comes from a spec. A
    flag without a spec raises `UnknownFlagError`, a subclass of `KeyError`
    and `ValueError`.

    A dataset builds its flag index from its `flag_specs`; a platform dataset
    collects the specs of all its interpreters and its area and border
    layers; a collection uses the union of the indexes of its children.

    The flags of a layer spec (`AreaFlagSpec`, `BorderFlagSpec`) come from
    the layers of the index: one flag per matching layer and variant value.

    Args:
        specs: Specs to add.
        layers: The area and border layers.

    Raises:
        AmbiguousOverrideError: Two specs give the same flag name.

    Examples:
        >>> index = FlagIndex([dataset.flag_index.get_spec(flag) for flag in ('Node.Model', 'Node.Results.price')])
        >>> print(index.get_unit('Node.Results.price'))
        EUR_per_MWh
        >>> index.get_topology_type('Node.Results.price')
        <TopologyTypeEnum.Node: 'Node'>
    """

    def __init__(self, specs: Iterable[FlagSpec] = (), layers: LayerSet = LayerSet()):
        self._specs: dict[str, FlagSpec] = {}
        self._spec_by_flag: dict[str, FlagSpec] = {}
        self._layers = layers
        self._references_checked = False
        for spec in specs:
            self.add_spec(spec)

    @classmethod
    def union(cls, indexes: Iterable[FlagIndex]) -> FlagIndex:
        """Build one index with the specs and layers of several indexes.

        Args:
            indexes: The indexes to combine.

        Returns:
            A new index.

        Raises:
            ValueError: Two indexes have specs with the same name that are not
                the same (`FlagSpec.same_as`).
            AmbiguousOverrideError: Two indexes have different layers with the
                same name.
        """
        indexes = tuple(indexes)
        union = cls(layers=LayerSet.union(index.layers for index in indexes))
        for index in indexes:
            for spec in index.specs:
                existing = union._specs.get(spec.name)
                if existing is None:
                    union.add_spec(spec)
                elif existing is not spec and not existing.same_as(spec):
                    raise ValueError(f'Two different specs for {spec.name!r}. Specs with one name must be the same.')
        return union

    @property
    def specs(self) -> tuple[FlagSpec, ...]:
        """All specs of this index."""
        return tuple(self._specs.values())

    @property
    def static_flags(self) -> set[str]:
        """Flag names of all specs except the flag families, whose values need a dataset."""
        return set(self._spec_by_flag)

    @property
    def parameter_names(self) -> set[str]:
        """Names of all parameters that a spec of this index declares."""
        return {d.name for spec in self._specs.values() for d in spec.params}

    @property
    def layers(self) -> LayerSet:
        """The area and border layers of this index."""
        return self._layers

    def add_spec(self, spec: FlagSpec) -> None:
        """Add a spec.

        Args:
            spec: The spec to add.

        Raises:
            AmbiguousOverrideError: Another spec gives one of the flag names.
        """
        existing = self._specs.get(spec.name)
        if existing is spec:
            return
        if existing is not None:
            raise AmbiguousOverrideError(f'Two specs declare {spec.name!r}.')
        flags = self._checked_flags_of(spec, self._spec_by_flag)
        self._specs[spec.name] = spec
        self._spec_by_flag.update(dict.fromkeys(flags, spec))
        self._references_checked = False

    def static_flags_of(self, spec: FlagSpec) -> list[str] | None:
        """Return the flags of a spec, or `None` for a flag family whose values need a dataset.

        The flags of a layer spec come from the layers of this index.
        """
        if isinstance(spec, LayerFlagSpec):
            return spec.expand_on_layers(layer.name for layer in self.get_layers(spec))
        return None if spec.is_family else spec.expand()

    def _checked_flags_of(self, spec: FlagSpec, spec_by_flag: dict[str, FlagSpec]) -> list[str]:
        flags = self.static_flags_of(spec) or []
        for flag in flags:
            if flag in spec_by_flag:
                raise AmbiguousOverrideError(
                    f'Specs {spec.name!r} and {spec_by_flag[flag].name!r} both give the flag {flag!r}.'
                )
        return flags

    def register_new_flag(self, flag: str, item_type: ItemTypeEnum, description: str = '', **fields: Any) -> FlagSpec:
        """Build a base spec of the class for `item_type` and add it.

        Args:
            flag: Flag name.
            item_type: Item type; selects `ModelFlagSpec`, `TimeSeriesFlagSpec` or `OtherFlagSpec`.
            description: Description of the flag.
            **fields: Further fields of the spec class, for example `unit`,
                `model_flag`, `index`, `topology_type` or `memberships`.

        Returns:
            The new spec.

        Examples:
            >>> from mesqual.enums import ItemTypeEnum, TopologyTypeEnum
            >>> index = FlagIndex([dataset.flag_index.get_spec('Node.Model')])
            >>> spec = index.register_new_flag(
            ...     'Generator.Model', ItemTypeEnum.Model, 'Generators.',
            ...     index='Generator', topology_type=TopologyTypeEnum.NodeConnectedElement,
            ...     memberships={'node': 'Node.Model'},
            ... )
        """
        spec = FlagSpec.class_for(item_type)(name=flag, description=description, **fields)
        self.add_spec(spec)
        return spec

    def get_spec_by_name(self, name: str) -> FlagSpec | None:
        """Return the spec with this name (a flag name or a template), or `None`."""
        return self._specs.get(name)

    def has_spec(self, flag: Any) -> bool:
        """`True` if a spec declares the flag."""
        try:
            self.get_spec(flag)
        except UnknownFlagError:
            return False
        return True

    def get_spec(self, flag: Any) -> FlagSpec:
        """Return the spec that declares the flag.

        Raises:
            UnknownFlagError: No spec declares the flag.
            AmbiguousOverrideError: Two flag family specs cover the flag.
        """
        flag = str(flag)
        spec = self._spec_by_flag.get(flag)
        if spec is not None:
            return spec
        covering = [s for s in self._specs.values() if s.is_family and self._family_covers(s, flag)]
        if len(covering) > 1:
            raise AmbiguousOverrideError(f'Specs {[s.name for s in covering]} all cover the flag {flag!r}.')
        if not covering:
            raise UnknownFlagError(f'No flag spec declares the flag {flag!r}.')
        return covering[0]

    def _family_covers(self, spec: FlagSpec, flag: str) -> bool:
        if not spec.covers(flag):
            return False
        if isinstance(spec, LayerFlagSpec):
            return spec.layer_name_of(flag) in {layer.name for layer in self.get_layers(spec)}
        return True

    def check_references(self, dataset: Dataset | None = None) -> None:
        """Check the references between the specs and layers of this index.

        - Each area layer has exactly one membership of its `base_model` to
          its `layer_model` (`get_mapping_column`).
        - The edge model of each border layer has `edge_ends` that refer to
          the base model of its area layer (`get_edge_ends`).
        - The base model of each layer spec is a `ModelFlagSpec` with a
          topology type that fits the spec class: `Node` or
          `NodeConnectedElement` for an area spec, `Edge` for a border spec.
        - Each `model_flag` of a time series or layer spec names a model
          spec: a `ModelFlagSpec`, or a layer spec on a model base (a layer
          model flag). A `model_flag` template is checked for each flag of
          the spec. The template of a flag family is checked only with a
          `dataset` that gives the family values.

        A platform dataset and a link collection run the check at
        construction, on their full index. Any other dataset runs it in its
        outermost fetch. Without a dataset, the result is kept until a spec
        is added.

        Args:
            dataset: The dataset that gives the values of the flag families.

        Raises:
            ValueError: A reference is not valid.
        """
        if self._references_checked and dataset is None:
            return
        for area_layer in self._layers.area_layers:
            self.get_mapping_column(area_layer)
        for border_layer in self._layers.border_layers:
            self.get_edge_ends(border_layer)
        for spec in self._specs.values():
            if isinstance(spec, LayerFlagSpec):
                self._check_layer_spec_base_model(spec)
            if getattr(spec, 'model_flag', None) is not None:
                self._check_model_flags(spec, self._flags_to_check(spec, dataset))
        self._references_checked = True

    def _flags_to_check(self, spec: FlagSpec, dataset: Dataset | None) -> list:
        flags = self.static_flags_of(spec)
        if flags is not None:
            return flags
        if not _has_placeholders(spec.model_flag):
            return [None]
        return [] if dataset is None else spec.expand(dataset)

    def _check_layer_spec_base_model(self, spec: LayerFlagSpec) -> None:
        if isinstance(spec.base, LayerFlagSpec):
            return
        model_flag = self._base_model_flag(spec)
        model_spec = self._spec_or_none(model_flag)
        allowed = [t.value for t in spec.BASE_TOPOLOGY_TYPES]
        if not isinstance(model_spec, ModelFlagSpec):
            raise ValueError(f'Spec {spec.name!r}: the base model {model_flag!r} names no ModelFlagSpec.')
        if model_spec.topology_type not in spec.BASE_TOPOLOGY_TYPES:
            raise ValueError(
                f'Spec {spec.name!r}: the base model {model_flag!r} has the topology type '
                f'{model_spec.topology_type.value}. Allowed: {allowed}.'
            )

    def _check_model_flags(self, spec: FlagSpec, flags: list) -> None:
        for flag in flags:
            model_flag = spec.model_flag if flag is None else self._fill_model_flag(spec, flag)
            if not self._is_model_spec(self._spec_or_none(model_flag)):
                raise ValueError(
                    f'Spec {spec.name!r}: model_flag {model_flag!r} names no ModelFlagSpec or layer model spec.'
                )

    @staticmethod
    def _is_model_spec(spec: FlagSpec | None) -> bool:
        return spec is not None and spec.item_type is ItemTypeEnum.Model

    @staticmethod
    def _fill_model_flag(spec: FlagSpec, flag: Any) -> str:
        if not _has_placeholders(spec.model_flag):
            return spec.model_flag
        return spec.model_flag.format(**spec.variant_of(flag).as_dict())

    def _spec_or_none(self, flag: Any) -> FlagSpec | None:
        try:
            return self.get_spec(flag)
        except UnknownFlagError:
            return None

    def get_item_type(self, flag: Any) -> ItemTypeEnum:
        """Return the item type of a flag. It follows from the class of its spec, or from the base of a layer spec."""
        return self.get_spec(flag).item_type

    def get_model_spec(self, flag: Any) -> FlagSpec | None:
        """Return the model spec of a flag: its own spec for a model flag, the spec of its model flag for a
        time series flag, else `None`.

        The model spec is a `ModelFlagSpec`, or a layer spec on a model base.
        """
        model_flag = self.get_linked_model_flag(flag)
        return None if model_flag is None else self.get_spec(model_flag)

    def get_linked_model_flag(self, flag: Any) -> str | None:
        """Return the model flag of a flag.

        A model flag returns itself, a time series flag its `model_flag`
        (can be `None`), any other flag `None`. A `model_flag` template is
        filled with the variant values of the flag, for example
        `'{area_layer}.Model'` gives `'Country.Model'`.

        A layer flag on a model base returns itself. Any other layer
        flag returns its filled `model_flag`, else the layer model flag:
        `layer_model` of an area layer, or the flag of the border spec on
        a model base for a border layer (`None` if no such spec exists).
        """
        spec = self.get_spec(flag)
        if spec.item_type is ItemTypeEnum.Model:
            return str(flag)
        if isinstance(spec, LayerFlagSpec):
            if spec.model_flag is not None:
                return self._fill_model_flag(spec, flag)
            return self.get_layer_model_flag(self.get_layer(flag))
        if isinstance(spec, TimeSeriesFlagSpec) and spec.model_flag is not None:
            return self._fill_model_flag(spec, flag)
        return getattr(spec, 'model_flag', None)

    def get_unit(self, flag: Any) -> Units.Unit | None:
        """Return the unit of a flag from its spec. A model flag has no unit (`None`)."""
        return getattr(self.get_spec(flag), 'unit', None)

    def get_fill_value(self, flag: Any) -> float | None:
        """Return the `fill_value` of a flag from its spec, `None` if the spec has none."""
        return getattr(self.get_spec(flag), 'fill_value', None)

    def get_quantity_type_enum(self, flag: Any) -> QuantityTypeEnum:
        """Return the quantity type of a flag.

        The spec value for an ambiguous unit, else the type of the unit.
        `UNKNOWN` for a flag without unit.
        """
        spec = self.get_spec(flag)
        quantity_type = getattr(spec, 'quantity_type', None)
        if quantity_type is not None:
            return quantity_type
        unit = getattr(spec, 'unit', None)
        return QuantityTypeEnum.UNKNOWN if unit is None else Units.get_quantity_type_enum(unit)

    def get_topology_type(self, flag: Any) -> TopologyTypeEnum:
        """Return the topology type of a flag.

        A model spec declares it; a time series spec takes it from the spec of
        its model flag; a layer spec has `Area` or `Border` from its class;
        any other flag, and a time series without model flag, has `Other`.
        """
        spec = self.get_spec(flag)
        if isinstance(spec, LayerFlagSpec):
            return spec.topology_type
        model_spec = self.get_model_spec(flag)
        return TopologyTypeEnum.Other if model_spec is None else model_spec.topology_type

    def get_visualization_type(self, flag: Any) -> VisualizationTypeEnum:
        """Return the visualization type: the one of the model spec, else the one of the topology type."""
        if not isinstance(self.get_spec(flag), LayerFlagSpec):
            model_spec = self.get_model_spec(flag)
            if model_spec is not None and model_spec.visualization_type is not None:
                return model_spec.visualization_type
        return self.get_topology_type(flag).visualization_type

    def get_schema(self, flag: Any) -> FrameSchema:
        """Return the index and column levels of the result of a flag.

        A spec schema wins. A model spec always has its schema
        (`ModelFlagSpec.schema`). Default for a time series spec: one index level
        `time` with role `time` and one column level, the object level of its
        model flag, with role `object`. Default for a layer spec: the schema
        of the base flag with the object level renamed to the `index` of the layer.

        Raises:
            ValueError: A time series schema without exactly one time level.
        """
        spec = self.get_spec(flag)
        if isinstance(spec, LayerFlagSpec):
            schema = self._layer_schema(spec, flag)
        else:
            schema = spec.declared_schema(flag)
            if schema is None:
                schema = self._default_schema(flag)
        if spec.item_type is ItemTypeEnum.TimeSeries and len(schema.time_levels) != 1:
            raise ValueError(f'Flag {flag!r}: a time series schema needs exactly one index level with role time.')
        return schema

    def _default_schema(self, flag: Any) -> FrameSchema:
        level = self.get_object_level(self.get_linked_model_flag(flag))
        return FrameSchema(index=(Level('time', 'time'),), columns=(Level(level, 'object'),))

    def _layer_schema(self, spec: LayerFlagSpec, flag: Any) -> FrameSchema:
        layer = self.get_layer(flag)
        declared = spec.declared_schema(flag, layer)
        if declared is not None:
            return declared
        base_schema = self.get_schema(self.get_base_flag(flag))
        return FrameSchema(
            index=_rename_object_levels(base_schema.index, layer.index),
            columns=_rename_object_levels(base_schema.columns, layer.index),
            dtype=base_schema.dtype,
        )

    def get_base_flag(self, flag: Any) -> str:
        """Return the base flag of a layer flag. A consumed dimension takes its first value.

        Raises:
            ValueError: The flag is not a layer flag, or a consumed dimension
                has values resolved per dataset.
        """
        spec = self._layer_spec(flag)
        defaults = {}
        for dimension in spec.base.variants:
            if dimension.name in spec.consumes:
                if dimension.is_resolved_per_dataset:
                    raise ValueError(
                        f'Spec {spec.name!r}: the consumed dimension {dimension.name!r} has values resolved per '
                        f'dataset, so the index cannot name a base flag for {flag!r}.'
                    )
                defaults[dimension.name] = next(iter(dimension.values_for(None)))
        return self.get_layer_variant(flag).base_flag(**defaults)

    def get_layer_variant(self, flag: Any) -> LayerVariantValues:
        """Return the variant values of a layer flag, with the layer of its base flag.

        For an area spec on a border base, the values also hold the border
        layer of the base (`border_layer`); for a border spec on an area
        base, the area layer of the base (`area_layer`).

        Raises:
            ValueError: The flag is not a layer flag.
        """
        spec = self._layer_spec(flag)
        values = spec.variant_of(flag).as_dict()
        base_layer = self.get_base_layer(spec, self._layers.get(values[spec.layer_placeholder]))
        if base_layer is not None:
            values[spec._base_layer_placeholder] = base_layer.name
        return LayerVariantValues(values, spec.base.name)

    def _layer_spec(self, flag: Any) -> LayerFlagSpec:
        spec = self.get_spec(flag)
        if not isinstance(spec, LayerFlagSpec):
            raise ValueError(f'{flag!r} is not a flag of a layer spec.')
        return spec

    def get_object_level(self, model_flag: Any) -> str:
        """Return the label of the object level of a model flag.

        The `index` of a `ModelFlagSpec`, or the `index` of the layer of a
        layer model flag.
        """
        spec = self.get_spec(model_flag)
        if isinstance(spec, LayerFlagSpec):
            return self.get_layer(model_flag).index
        return spec.index

    def get_memberships(self, model_flag: Any) -> dict[str, str]:
        """Return the membership columns of a model flag: column to the model flag it refers to.

        Raises:
            ValueError: The flag is not declared by a `ModelFlagSpec`.
        """
        spec = self.get_spec(model_flag)
        if not isinstance(spec, ModelFlagSpec):
            raise ValueError(f'{model_flag!r} is not a model flag with memberships.')
        return dict(spec.memberships)

    def get_referencing_columns(self, model_flag: Any) -> list[tuple[str, str]]:
        """Return all `(model flag, column)` pairs whose membership refers to `model_flag`.

        Examples:
            >>> dataset.flag_index.get_referencing_columns('Node.Model')
            [('Line.Model', 'node_from'), ('Line.Model', 'node_to'), ('Generator.Model', 'node'), ('Demand.Model', 'node')]
        """
        target = str(model_flag)
        pairs = []
        for spec in self._specs.values():
            if isinstance(spec, ModelFlagSpec):
                for flag in spec.expand() if not spec.is_family else []:
                    pairs += [(flag, column) for column, ref in spec.memberships.items() if ref == target]
        return pairs

    def get_mapping_column(self, layer: AreaLayer) -> str:
        """Return the membership column of `layer.base_model` whose target is `layer.layer_model`.

        Raises:
            ValueError: The base model has no `ModelFlagSpec`, or not exactly
                one membership to the layer model.
        """
        if not isinstance(self._spec_or_none(layer.base_model), ModelFlagSpec):
            raise ValueError(f'Area layer {layer.name!r}: base_model {layer.base_model!r} names no ModelFlagSpec.')
        columns = [c for c, target in self.get_memberships(layer.base_model).items() if target == layer.layer_model]
        if len(columns) != 1:
            raise ValueError(
                f'Area layer {layer.name!r}: {layer.base_model!r} needs exactly one membership to '
                f'{layer.layer_model!r}, found {columns}.'
            )
        return columns[0]

    def get_edge_ends(self, layer: BorderLayer) -> tuple[str, str]:
        """Return the `edge_ends` of the edge model of a border layer.

        Raises:
            ValueError: The edge model has no `edge_ends`, or its ends do not
                refer to the base model of the area layer.
        """
        spec = self._spec_or_none(layer.edge_model)
        if not isinstance(spec, ModelFlagSpec) or spec.edge_ends is None:
            raise ValueError(f'Border layer {layer.name!r}: edge model {layer.edge_model!r} declares no edge_ends.')
        base_model = self._layers.area_layer_of(layer).base_model
        if spec.memberships[spec.edge_ends[0]] != base_model:
            raise ValueError(
                f'Border layer {layer.name!r}: the edge_ends of {layer.edge_model!r} refer to '
                f'{spec.memberships[spec.edge_ends[0]]!r}, not to the base model {base_model!r} of the area layer.'
            )
        return spec.edge_ends

    def get_layers(self, spec: FlagSpec) -> list[AreaLayer | BorderLayer]:
        """Return the layers of a layer spec. Any other spec has none.

        - `AreaFlagSpec` on a node base: the area layers whose `base_model`
          is the model flag of the base.
        - `AreaFlagSpec` on a border base: the area layers with exactly one
          border layer among the layers of the base.
        - `BorderFlagSpec` on an edge base: the border layers whose
          `edge_model` is the model flag of the base.
        - `BorderFlagSpec` on an area base: the border layers whose area
          layer is a layer of the base.

        Raises:
            ValueError: An area layer has two border layers among the layers
                of a border base.
        """
        if isinstance(spec, AreaFlagSpec) and spec._has_layer_base:
            return [area for area in self._layers.area_layers if self._border_layer_for(spec, area) is not None]
        if isinstance(spec, AreaFlagSpec):
            return [layer for layer in self._layers.area_layers if layer.base_model == self._base_model_flag(spec)]
        if isinstance(spec, BorderFlagSpec) and spec._has_layer_base:
            names = {layer.name for layer in self.get_layers(spec.base)}
            return [layer for layer in self._layers.border_layers if layer.area_layer in names]
        if isinstance(spec, BorderFlagSpec):
            return [layer for layer in self._layers.border_layers if layer.edge_model == self._base_model_flag(spec)]
        return []

    def _border_layer_for(self, spec: AreaFlagSpec, area_layer: AreaLayer) -> BorderLayer | None:
        matches = [layer for layer in self.get_layers(spec.base) if layer.area_layer == area_layer.name]
        if len(matches) > 1:
            raise ValueError(
                f'Spec {spec.name!r}: area layer {area_layer.name!r} has two border layers among the layers of the '
                f'base: {[m.name for m in matches]}.'
            )
        return matches[0] if matches else None

    @staticmethod
    def _base_model_flag(spec: LayerFlagSpec) -> str:
        base = spec.base
        return base.name if isinstance(base, ModelFlagSpec) else base.model_flag

    def get_layer(self, flag: Any) -> AreaLayer | BorderLayer:
        """Return the layer of a layer flag.

        Raises:
            ValueError: The flag is not a layer flag.
        """
        return self._layers.get(self._layer_spec(flag).layer_name_of(flag))

    def get_base_layer(self, spec: LayerFlagSpec, layer: AreaLayer | BorderLayer) -> AreaLayer | BorderLayer | None:
        """Return the layer of the base flag of a layer flag on `layer`.

        The border layer of an area layer for an area spec on a border base,
        the area layer of a border layer for a border spec on an area base,
        else `None`.
        """
        if isinstance(spec, AreaFlagSpec) and spec._has_layer_base:
            return self._border_layer_for(spec, layer)
        if isinstance(spec, BorderFlagSpec) and spec._has_layer_base:
            return self._layers.area_layer_of(layer)
        return None

    def get_layer_model_flag(self, layer: AreaLayer | BorderLayer) -> str | None:
        """Return the model flag of the areas or borders of a layer.

        `layer_model` of an area layer. For a border layer, the flag of the
        border spec on a model base (for example `BorderModel()`) on that
        layer, or `None` if the index has no such spec.
        """
        if isinstance(layer, AreaLayer):
            return layer.layer_model
        return self._border_model_flag(layer)

    def _border_model_flag(self, layer: BorderLayer) -> str | None:
        for spec in self._specs.values():
            if isinstance(spec, BorderFlagSpec) and spec.is_model and layer in self.get_layers(spec):
                return spec.name.format(**{spec.layer_placeholder: layer.name})
        return None

    def get_all_timeseries_flags_for_model_flag(self, dataset: Dataset, flag: Any) -> set[str]:
        """Return the accepted time series flags of `dataset` whose model flag is `flag`."""
        return {
            f
            for f in dataset.accepted_flags
            if self.get_item_type(f) == ItemTypeEnum.TimeSeries and self.get_linked_model_flag(f) == flag
        }

    def get_flag_from_string(self, flag_string: str) -> str:
        """Return the flag for a string. Flags are strings."""
        return flag_string


def _has_placeholders(name: str) -> bool:
    return any(field is not None for _, field, _, _ in string.Formatter().parse(name))


def _rename_object_levels(levels: tuple[Level, ...], name: str) -> tuple[Level, ...]:
    return tuple(Level(name, 'object') if level.role == 'object' else level for level in levels)
