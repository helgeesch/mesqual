"""Layer flag specs: a base spec and an aggregation to area or border layers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Hashable, Iterable, Mapping

from mesqual.enums import ItemTypeEnum, QuantityTypeEnum, TopologyTypeEnum
from mesqual.flag.dimension import REQUIRED, Dimension, VariantValues
from mesqual.flag.flag_spec import FlagSpec, ModelFlagSpec, TimeSeriesFlagSpec
from mesqual.flag.layers import AreaLayer, BorderLayer
from mesqual.flag.schema import FrameSchema
from mesqual.units import Units

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset
    from mesqual.layer_flags.aggregations import AreaAggregation, BorderAggregation


class _LayerValues:
    """Values function of the layer dimension of a layer spec: the matching layers of the dataset.

    It takes a dataset instance, or a dataset class with its registered layers.
    """

    def __init__(self, spec: LayerFlagSpec):
        self._spec = spec

    def __call__(self, dataset: Dataset | type[Dataset]) -> dict[str, str]:
        index = dataset._class_flag_index() if isinstance(dataset, type) else dataset.flag_index
        return {layer.name: layer.description for layer in index.get_layers(self._spec)}


@dataclass(frozen=True, eq=False)
class LayerFlagSpec(FlagSpec):
    """Common fields of `AreaFlagSpec` and `BorderFlagSpec`. Not used directly.

    A layer spec names a base spec and an aggregation. Item type, unit,
    quantity type and `fill_value` come from the base, the topology type
    from the class.

    The layer is a variant dimension named `layer_placeholder`, resolved per
    dataset from the registered layers. The other variant dimensions are the
    ones of the base, without `consumes`. `params` holds the base parameters
    followed by the parameters of the aggregation (`aggregation_params`).

    Attributes:
        base: The base spec.
        aggregation: The aggregation to the layer.
        model_flag: Template of the linked model flag. `None`: the layer
            model flag.
        schema: Index and column levels of the result, or a function of the
            variant values and the layer. `None`: the base schema with the
            object level renamed to the `index` of the layer.
        consumes: Variant dimensions of the base that the aggregation
            resolves itself. They are not in the spec name.
        layer_placeholder: Name of the layer dimension and of its
            placeholder in `name`.
        aggregation_params: Parameters of the aggregation, given as
            `params`. The interpreter passes them to `aggregate`, not to
            the base fetch.
    """

    topology_type: ClassVar[TopologyTypeEnum]
    BASE_TOPOLOGY_TYPES: ClassVar[tuple[TopologyTypeEnum, ...]]
    LAYER_KIND: ClassVar[str]

    base: FlagSpec = REQUIRED
    aggregation: Any = REQUIRED
    model_flag: str | None = None
    schema: FrameSchema | Callable[[VariantValues, AreaLayer | BorderLayer], FrameSchema] | None = None
    consumes: tuple[str, ...] = ()
    layer_placeholder: str = REQUIRED
    variants: tuple[Dimension, ...] = field(default=(), init=False)
    aggregation_params: tuple[Dimension, ...] = field(default=(), init=False)

    def __post_init__(self):
        self._check_required_fields()
        self._check_base()
        object.__setattr__(self, 'consumes', tuple(self.consumes))
        base_dimensions = {d.name for d in self.base.dimensions}
        unknown = set(self.consumes) - {d.name for d in self.base.variants}
        if unknown:
            raise ValueError(f'Spec {self.name!r}: consumes {sorted(unknown)}, which are no variants of the base.')
        if self.layer_placeholder in base_dimensions:
            raise ValueError(
                f'Spec {self.name!r}: the layer placeholder {self.layer_placeholder!r} is a dimension of the base. '
                f'Set another layer_placeholder.'
            )
        layer_dimension = Dimension(
            self.layer_placeholder, f'{self.LAYER_KIND.capitalize()} layer.', _LayerValues(self)
        )
        resolved_here = (*self.consumes, self._base_layer_placeholder)
        kept = tuple(d for d in self.base.variants if d.name not in resolved_here)
        object.__setattr__(self, 'aggregation_params', tuple(self.params))
        object.__setattr__(self, 'params', self.base.params + self.aggregation_params)
        object.__setattr__(self, 'variants', (layer_dimension,) + kept)
        super().__post_init__()
        self._check_model_flag_placeholders()
        self.aggregation.check_spec(self)

    def _check_base(self) -> None:
        raise NotImplementedError

    def on_base(self, base: FlagSpec) -> LayerFlagSpec:
        """Return this spec built on another base: the parameters of `base`, then the aggregation parameters."""
        return replace(self, base=base, params=self.aggregation_params)

    @property
    def _base_layer_placeholder(self) -> str | None:
        """The layer dimension of a layer base, else `None`.

        A layer base is a border spec for an area spec, an area spec for a
        border spec. The interpreter resolves its layer dimension from the
        layer of the flag.
        """
        return self.base.layer_placeholder if isinstance(self.base, LayerFlagSpec) else None

    @property
    def _has_layer_base(self) -> bool:
        """`True` if the base is a layer spec (`_base_layer_placeholder`)."""
        return self._base_layer_placeholder is not None

    @property
    def item_type(self) -> ItemTypeEnum:
        """Item type of the base."""
        return self.base.item_type

    @property
    def unit(self) -> Units.Unit | None:
        """Unit of the base. `None` for a model base."""
        return getattr(self.base, 'unit', None)

    @property
    def quantity_type(self) -> QuantityTypeEnum | None:
        """Quantity type of the base, set only for an ambiguous unit."""
        return getattr(self.base, 'quantity_type', None)

    @property
    def fill_value(self) -> float | None:
        """`fill_value` of the base."""
        return getattr(self.base, 'fill_value', None)

    @property
    def visualization_type(self) -> None:
        """`None`: the visualization type follows from the topology type."""
        return None

    @property
    def is_model(self) -> bool:
        """`True` if the base is a model flag, so the layer flags are the layer model flags."""
        return self.base.item_type is ItemTypeEnum.Model

    @property
    def layer_dimension(self) -> Dimension:
        """The layer dimension."""
        return self.variants[0]

    def layer_name_of(self, flag: Any) -> str:
        """Return the layer name of a flag of this spec."""
        return getattr(self.variant_of(flag), self.layer_placeholder)

    def expand_on_layers(self, layer_names: Iterable[str]) -> list[str] | None:
        """Return the flag names for the given layers, or `None` if another variant dimension needs a dataset."""
        if any(d.is_resolved_per_dataset for d in self.variants[1:]):
            return None
        names = [d.name for d in self.variants]
        value_lists = [list(layer_names)] + [list(d.values) for d in self.variants[1:]]
        return [self._fill(dict(zip(names, values))) for values in itertools.product(*value_lists)]

    def declared_schema(self, flag: Any, layer: AreaLayer | BorderLayer | None = None) -> FrameSchema | None:
        """Return the declared schema for a flag, or `None` for the default schema.

        Raises:
            ValueError: The schema is a function and `layer` is `None`.
        """
        if not callable(self.schema):
            return self.schema
        if layer is None:
            raise ValueError(f'Spec {self.name!r}: the schema is a function of the layer. Pass the layer.')
        return self.schema(self.variant_of(flag), layer)

    def flag_description(self, flag: Any, dataset: Dataset | None = None) -> str:
        """Return the description of one flag.

        The spec description, the base description, the layer description
        (with a dataset only) and one line per other variant.
        """
        variant = self.variant_of(flag)
        parts = [self.description, f'Base: {self.base.description}']
        if dataset is not None:
            parts.append(f'Layer: {self.layer_dimension.describe_value(self.layer_name_of(flag), dataset)}')
        for dimension in self.variants[1:]:
            label = dimension.name.replace('_', ' ').capitalize()
            parts.append(f'{label}: {dimension.describe_value(getattr(variant, dimension.name), dataset)}.')
        return ' '.join(parts)

    def same_as(self, other: FlagSpec) -> bool:
        """`True` if both specs declare the same thing.

        The class, name, version, `model_flag`, `consumes`, `layer_placeholder`
        and the aggregation parameters are equal, the base specs are the same
        and the aggregations are equal (same type and fields).
        """
        if type(self) is not type(other) or self._identity() != other._identity():
            return False
        own, theirs = self.aggregation_params, other.aggregation_params
        return (
            self.base.same_as(other.base) and len(own) == len(theirs) and all(a.same_as(b) for a, b in zip(own, theirs))
        )

    def _class_identity(self) -> tuple:
        return self.aggregation, self.model_flag, self.consumes, self.layer_placeholder

    def _check_base_model_flag(self) -> None:
        if isinstance(self.base, TimeSeriesFlagSpec) and self.base.model_flag is None:
            raise ValueError(f'Spec {self.name!r}: base {self.base.name!r} has no model flag.')


@dataclass(frozen=True, eq=False)
class AreaFlagSpec(LayerFlagSpec):
    """A variable on area layers: a base spec and an area aggregation.

    The base is a node or node-connected spec (a model spec with topology
    type `Node` or `NodeConnectedElement`, or a time series spec on such a
    model), or a border spec. On a node base, the layers are the area layers
    whose `base_model` is the model flag of the base. On a border base, the
    layers are the area layers with exactly one border layer among the
    layers of the base. `FlagIndex.check_references` checks the topology
    type of the base model.

    Attributes:
        aggregation: An `AreaAggregation`.
        layer_placeholder: Name of the layer dimension. Default `'area_layer'`.

    Examples:
        >>> from mesqual.layer_flags import WeightedMean
        >>> AREA_PRICE_SPEC = AreaFlagSpec(
        ...     name='{area_layer}.Results.price',
        ...     base=dataset.flag_index.get_spec('Node.Results.price'),
        ...     aggregation=WeightedMean(weight='Node.Results.demand_total'),
        ...     description='Mean of the node prices of the area, weighted by the total node demand.',
        ... )
    """

    topology_type: ClassVar[TopologyTypeEnum] = TopologyTypeEnum.Area
    BASE_TOPOLOGY_TYPES: ClassVar[tuple[TopologyTypeEnum, ...]] = (
        TopologyTypeEnum.Node,
        TopologyTypeEnum.NodeConnectedElement,
    )
    LAYER_KIND: ClassVar[str] = 'area'

    aggregation: AreaAggregation = REQUIRED
    layer_placeholder: str = 'area_layer'

    def _check_base(self) -> None:
        if not isinstance(self.base, (ModelFlagSpec, TimeSeriesFlagSpec, BorderFlagSpec)):
            raise TypeError(f'Spec {self.name!r}: the base must be a model, time series or border spec.')
        self._check_base_model_flag()


@dataclass(frozen=True, eq=False)
class BorderFlagSpec(LayerFlagSpec):
    """A variable on border layers: a base spec and a border aggregation.

    The base is an edge spec (a model spec with topology type `Edge`, or a
    time series spec on such a model), or an area spec. On an edge base,
    the layers are the border layers whose `edge_model` is the model flag of
    the base. On an area base, the layers are the border layers whose area
    layer is a layer of the base. `FlagIndex.check_references` checks the
    topology type of the base model.

    Attributes:
        aggregation: A `BorderAggregation`.
        layer_placeholder: Name of the layer dimension. Default `'border_layer'`.

    Examples:
        >>> from mesqual.layer_flags import BorderFlowSum
        >>> BORDER_FLOW_SPEC = BorderFlagSpec(
        ...     name='{border_layer}.Results.flow_{side}_{direction}',
        ...     base=dataset.flag_index.get_spec('Line.Results.flow_sent_net'),
        ...     aggregation=BorderFlowSum(),
        ...     description='Sum of the line flows across the border, in border direction.',
        ... )
    """

    topology_type: ClassVar[TopologyTypeEnum] = TopologyTypeEnum.Border
    BASE_TOPOLOGY_TYPES: ClassVar[tuple[TopologyTypeEnum, ...]] = (TopologyTypeEnum.Edge,)
    LAYER_KIND: ClassVar[str] = 'border'

    aggregation: BorderAggregation = REQUIRED
    layer_placeholder: str = 'border_layer'

    def _check_base(self) -> None:
        if not isinstance(self.base, (ModelFlagSpec, TimeSeriesFlagSpec, AreaFlagSpec)):
            raise TypeError(f'Spec {self.name!r}: the base must be a model, time series or area spec.')
        self._check_base_model_flag()


def area_flag_specs(
    bases: Iterable[FlagSpec],
    aggregation: AreaAggregation,
    name: Callable[[FlagSpec], str],
    description: str | Callable[[FlagSpec], str],
    **fields: Any,
) -> tuple[AreaFlagSpec, ...]:
    """Build one `AreaFlagSpec` per base spec with the same aggregation.

    The platform still lists the result explicitly on an interpreter.

    Args:
        bases: The base specs.
        aggregation: The area aggregation of all specs.
        name: Function of the base spec that returns the name of the area spec.
        description: Description of all specs, or a function of the base spec.
        **fields: Further fields of `AreaFlagSpec`, for example `params`.

    Examples:
        >>> from mesqual.layer_flags import Sum
        >>> AREA_SUM_SPECS = area_flag_specs(
        ...     bases=tuple(dataset.flag_index.get_spec(flag) for flag in ('Node.Results.demand_total', 'Node.Results.generation_gas')),
        ...     aggregation=Sum(),
        ...     name=lambda base: base.name.replace('Node.', '{area_layer}.'),
        ...     description='Sum over the nodes of the area.',
        ... )
        >>> [spec.name for spec in AREA_SUM_SPECS]
        ['{area_layer}.Results.demand_{product}', '{area_layer}.Results.generation_{fuel}']
    """
    return tuple(
        AreaFlagSpec(
            name=name(base),
            base=base,
            aggregation=aggregation,
            description=description(base) if callable(description) else description,
            **fields,
        )
        for base in bases
    )


class LayerVariantValues(VariantValues):
    """Variant values of a layer flag, as an aggregation gets them.

    Besides the variant values of the flag, it holds the layer placeholder
    of the base for an area spec on a border base (`border_layer`) and for a
    border spec on an area base (`area_layer`). `base_flag` builds base flag
    names, so an aggregation never needs the flag grammar of the platform.
    `FlagIndex.get_layer_variant` builds it.

    Args:
        values: The variant values, with the layer of the base.
        base_name: The name of the base spec.

    Examples:
        >>> variant = dataset.flag_index.get_layer_variant('CountryBorder.Results.flow_sent_up')
        >>> variant.direction
        'up'
        >>> variant.base_flag(direction='down')
        'Line.Results.flow_sent_down'
    """

    __slots__ = ('_base_name',)

    def __init__(self, values: Mapping[str, Hashable], base_name: str):
        super().__init__(values)
        object.__setattr__(self, '_base_name', base_name)

    def base_flag(self, **changes: Hashable) -> str:
        """Return the base flag for these variant values, with some values changed.

        Args:
            **changes: Values of base dimensions to change, and the values
                of the dimensions that the spec consumes.

        Raises:
            ValueError: A dimension of the base name has no value.
        """
        try:
            return self._base_name.format(**{**self.as_dict(), **changes})
        except KeyError as missing:
            raise ValueError(f'Base flag {self._base_name!r} needs a value for {missing}.') from None
