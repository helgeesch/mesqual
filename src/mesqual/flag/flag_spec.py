"""Flag specs: the declaration of one variable, the single source of truth for its flags."""

from __future__ import annotations

import dataclasses
import itertools
import re
import string
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Hashable, Mapping

from mesqual.enums import ItemTypeEnum, QuantityTypeEnum, TopologyTypeEnum, VisualizationTypeEnum
from mesqual.flag.dimension import REQUIRED, Dimension, VariantValues
from mesqual.flag.schema import FrameSchema, Level
from mesqual.units import Units

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset


@dataclass(frozen=True, eq=False)
class FlagSpec:
    """Common fields of the base specs `ModelFlagSpec`, `TimeSeriesFlagSpec` and `OtherFlagSpec`.

    Not used directly. The item type follows from the class.

    A spec name is a flag name, or a template with one `{placeholder}` per
    variant dimension. mesqual only fills in the placeholders; it never parses
    flag names. Everything mesqual needs to know about a flag comes from the
    fields of its spec.

    Specs use identity for `==` and `hash`, because they can hold functions.
    `same_as` compares the content.

    Attributes:
        name: Flag name, or a template with one `{placeholder}` per variant dimension.
        description: What the variable is, how it is computed, what to watch for.
        variants: Variant dimensions. Each value combination is a flag of its own.
        params: Parameter dimensions. A setting of the same flag, resolved at fetch.
        version: Bump by hand when the logic changes. Part of the version identity.
        schema: Index and column levels of the result, a function of the
            variant values that returns them, or `None` for the default
            schema. Every spec class provides it, as a field or as a
            read-only property.
    """

    item_type: ClassVar[ItemTypeEnum]

    name: str
    description: str
    variants: tuple[Dimension, ...] = ()
    params: tuple[Dimension, ...] = ()
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'variants', tuple(self.variants))
        object.__setattr__(self, 'params', tuple(self.params))
        self._check_required_fields()
        self._check_dimensions()
        self._check_placeholders()
        if not self.is_family:
            _ = self._variants_by_flag

    @property
    def schema(self) -> FrameSchema | Callable[[VariantValues], FrameSchema] | None:
        """Index and column levels of the result. Each spec class overrides it.

        Raises:
            NotImplementedError: The spec class does not provide a schema.
        """
        raise NotImplementedError(f'{type(self).__name__} does not provide a schema.')

    @property
    def is_family(self) -> bool:
        """`True` if a variant dimension resolves its values per dataset."""
        return any(d.is_resolved_per_dataset for d in self.variants)

    @property
    def dimensions(self) -> tuple[Dimension, ...]:
        """Variant and parameter dimensions, in this order."""
        return self.variants + self.params

    def expand(self, dataset: Dataset | None = None) -> list[str]:
        """Return the flag names of this spec.

        Args:
            dataset: Dataset that resolves variant values given as a function.
                Not needed for a spec without such values.

        Returns:
            One flag name per combination of variant values.

        Raises:
            ValueError: The spec is a flag family and `dataset` is `None`.
        """
        if not self.is_family:
            return list(self._variants_by_flag)
        return [self._fill(combination) for combination in self._combinations(dataset)]

    def static_flags(self) -> set[str]:
        """The flags of the spec, or an empty set for a flag family, whose values need a dataset."""
        return set() if self.is_family else set(self.expand())

    def covers(self, flag: Any) -> bool:
        """`True` if `flag` is one of the flags of this spec.

        For a flag family, the values of the family dimensions are not checked;
        that needs the dataset.
        """
        try:
            self.variant_of(flag)
        except KeyError:
            return False
        return True

    def variant_of(self, flag: Any) -> VariantValues:
        """Return the variant values of a flag of this spec.

        Args:
            flag: A flag of this spec.

        Returns:
            One attribute per variant dimension.

        Raises:
            KeyError: The spec does not cover the flag.

        Examples:
            >>> flow_spec = dataset.flag_index.get_spec('Line.Results.flow_sent_net')
            >>> flow_spec.variant_of('Line.Results.flow_sent_net').direction
            'net'
        """
        flag = str(flag)
        if not self.is_family:
            try:
                return self._variants_by_flag[flag]
            except KeyError:
                raise KeyError(f'Spec {self.name!r} does not cover flag {flag!r}.') from None
        match = self._family_pattern.fullmatch(flag)
        if match is None:
            raise KeyError(f'Spec {self.name!r} does not cover flag {flag!r}.')
        return VariantValues({d.name: self._static_value(d, match.group(d.name)) for d in self.variants})

    def check_variant_values(self, flag: Any, dataset: Dataset | None = None) -> None:
        """Raise `ValueError` if a family value of `flag` is not a value on `dataset`.

        Args:
            flag: A flag of this spec.
            dataset: Dataset that resolves the family values.

        Raises:
            FlagUnavailableError: The values function of a variant fetches an
                unavailable flag, as in `Dimension.check_value`.
            ValueError: A family value is not a value on `dataset`.
        """
        variant = self.variant_of(flag)
        for dimension in self.variants:
            if dimension.is_resolved_per_dataset:
                dimension.check_value(getattr(variant, dimension.name), dataset)

    def resolve_params(self, given: Mapping[str, Any] | None = None, dataset: Dataset | None = None) -> dict[str, Any]:
        """Return the resolved flag parameters: per parameter the given value, else its default, then the check.

        Each parameter resolves with `Dimension.resolve`. A config default, for
        example `param_defaults` of a dataset config, is not part of the spec:
        the caller adds it to `given`. The variant values of a flag family are
        checked with `check_variant_values`.

        Args:
            given: Given values by parameter name. `None`: no values.
            dataset: Dataset that resolves values given as a function. With
                `None` and such values, neither the values nor `validate`
                are checked.

        Returns:
            One value per parameter of the spec, in the order of `params`.

        Raises:
            FlagUnavailableError: The values function of a parameter fetches
                an unavailable flag, as in `Dimension.check_value`.
            ValueError: `given` names a parameter that the spec does not
                declare, a `REQUIRED` parameter has no value, or a value is
                not allowed.

        Examples:
            The node price has the parameter `solution` (values `'first'` and
            `'final'`, default `'final'`):

            >>> price_spec = dataset.flag_index.get_spec('Node.Results.price')
            >>> price_spec.resolve_params()
            {'solution': 'final'}
            >>> price_spec.resolve_params({'solution': 'first'})
            {'solution': 'first'}
            >>> price_spec.resolve_params({'solution': 'best'})
            Traceback (most recent call last):
            ValueError: 'best' is not a value of 'solution'. Allowed: ['first', 'final'].
            >>> price_spec.resolve_params({'scenario': 'low'})
            Traceback (most recent call last):
            ValueError: Spec 'Node.Results.price' declares no parameters ['scenario']. Parameters: ['solution'].
        """
        given = given or {}
        names = [d.name for d in self.params]
        unknown = set(given) - set(names)
        if unknown:
            raise ValueError(f'Spec {self.name!r} declares no parameters {sorted(unknown)}. Parameters: {names}.')
        return {d.name: d.resolve(given, dataset) for d in self.params}

    def flag_description(self, flag: Any, dataset: Dataset | None = None) -> str:
        """Return the description of one flag: the spec description and one line per variant.

        Examples:
            >>> flow_spec = dataset.flag_index.get_spec('Line.Results.flow_received_net')
            >>> flow_spec.flag_description('Line.Results.flow_received_net')
            'Power flow on the line. Side: where the flow leaves the line, after losses. Direction: up minus down.'
        """
        variant = self.variant_of(flag)
        parts = [self.description]
        for dimension in self.variants:
            label = dimension.name.replace('_', ' ').capitalize()
            parts.append(f'{label}: {dimension.describe_value(getattr(variant, dimension.name), dataset)}.')
        return ' '.join(parts)

    def same_as(self, other: FlagSpec) -> bool:
        """`True` if both specs declare the same thing.

        Two specs are the same when their class, name, dimensions (names,
        values and defaults), version and the fields of their class are equal.
        """
        if type(self) is not type(other) or self._identity() != other._identity():
            return False
        return all(a.same_as(b) for a, b in zip(self.dimensions, other.dimensions)) and (
            len(self.dimensions) == len(other.dimensions)
        )

    @classmethod
    def class_for(cls, item_type: ItemTypeEnum) -> type[FlagSpec]:
        """Return the base spec class for an item type.

        Examples:
            >>> from mesqual.enums import ItemTypeEnum
            >>> FlagSpec.class_for(ItemTypeEnum.Model)
            <class 'mesqual.flag.flag_spec.ModelFlagSpec'>
        """
        match item_type:
            case ItemTypeEnum.Model:
                return ModelFlagSpec
            case ItemTypeEnum.TimeSeries:
                return TimeSeriesFlagSpec
            case ItemTypeEnum.Other:
                return OtherFlagSpec
        raise ValueError(f'No spec class for item type {item_type!r}.')

    def declared_schema(self, flag: Any) -> FrameSchema | None:
        """Return the schema that the spec declares for a flag, or `None` for the default schema."""
        schema = self.schema
        if callable(schema):
            return schema(self.variant_of(flag))
        return schema

    def _identity(self) -> tuple:
        return (self.name, self.version) + tuple(self._class_identity())

    def _class_identity(self) -> tuple:
        return ()

    def _check_required_fields(self) -> None:
        for f in dataclasses.fields(self):
            if getattr(self, f.name) is REQUIRED:
                raise TypeError(f'{type(self).__name__} {self.name!r}: field {f.name!r} is required.')

    def _check_dimensions(self) -> None:
        names = [d.name for d in self.dimensions]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f'Spec {self.name!r}: dimension names must be unique, got {duplicates} twice.')
        for dimension in self.variants:
            if dimension.values is None:
                raise ValueError(f'Spec {self.name!r}: variant {dimension.name!r} needs values.')
            if dimension.has_default or dimension.validate is not None:
                raise ValueError(f'Spec {self.name!r}: variant {dimension.name!r} takes no default or validate.')

    def _check_model_flag_placeholders(self) -> None:
        """Raise `ValueError` if a placeholder of a `model_flag` template is no variant dimension."""
        model_flag = getattr(self, 'model_flag', None)
        if model_flag is None:
            return
        placeholders = {name for _, name, _, _ in string.Formatter().parse(model_flag) if name is not None}
        unknown = placeholders - {d.name for d in self.variants}
        if unknown:
            raise ValueError(
                f'Spec {self.name!r}: model_flag {model_flag!r} has placeholders {sorted(unknown)} '
                f'that are no variant dimensions.'
            )

    def _check_placeholders(self) -> None:
        placeholders = [name for _, name, _, _ in string.Formatter().parse(self.name) if name is not None]
        expected = sorted(d.name for d in self.variants)
        if sorted(placeholders) != expected:
            raise ValueError(
                f'Spec {self.name!r}: placeholders {sorted(placeholders)} must match the variant dimensions {expected}.'
            )

    @cached_property
    def _variants_by_flag(self) -> Mapping[str, VariantValues]:
        variants_by_flag = {}
        for combination in self._combinations(None):
            flag = self._fill(combination)
            if flag in variants_by_flag:
                raise ValueError(f'Spec {self.name!r}: two variant combinations give the flag {flag!r}.')
            variants_by_flag[flag] = VariantValues(combination)
        return MappingProxyType(variants_by_flag)

    def _combinations(self, dataset: Dataset | None) -> list[dict[str, Hashable]]:
        names = [d.name for d in self.variants]
        value_lists = [list(d.values_for(dataset)) for d in self.variants]
        return [dict(zip(names, values)) for values in itertools.product(*value_lists)]

    def _fill(self, combination: Mapping[str, Hashable]) -> str:
        return self.name.format(**combination)

    @cached_property
    def _family_pattern(self) -> re.Pattern:
        dimensions = {d.name: d for d in self.variants}
        pattern = ''
        for literal, name, _, _ in string.Formatter().parse(self.name):
            pattern += re.escape(literal)
            if name is None:
                continue
            dimension = dimensions[name]
            if dimension.is_resolved_per_dataset:
                pattern += f'(?P<{name}>.+?)'
            else:
                options = sorted((re.escape(str(v)) for v in dimension.values), key=len, reverse=True)
                pattern += f'(?P<{name}>{"|".join(options)})'
        return re.compile(pattern)

    @staticmethod
    def _static_value(dimension: Dimension, text: str) -> Hashable:
        if dimension.is_resolved_per_dataset:
            return text
        return next(v for v in dimension.values if str(v) == text)


@dataclass(frozen=True, eq=False)
class ModelFlagSpec(FlagSpec):
    """The model table of one object type: one row per object.

    Attributes:
        index: Name of the index level, for example `'Node'`. Also the column
            level name of the linked time series.
        topology_type: Role of the objects in the network topology.
        memberships: Column to the model flag it refers to. The single source
            of object relations; mesqual does not detect them from column names.
        edge_ends: Topology type `Edge` only: the two memberships that hold
            the ends, in line direction.
        visualization_type: `None`: follows from the topology type.

    Examples:
        >>> from mesqual.enums import TopologyTypeEnum
        >>> LINE_MODEL_SPEC = ModelFlagSpec(
        ...     name='Line.Model',
        ...     index='Line',
        ...     topology_type=TopologyTypeEnum.Edge,
        ...     description='Lines between two nodes.',
        ...     memberships={'node_from': 'Node.Model', 'node_to': 'Node.Model'},
        ...     edge_ends=('node_from', 'node_to'),
        ... )
    """

    item_type: ClassVar[ItemTypeEnum] = ItemTypeEnum.Model

    index: str = REQUIRED
    topology_type: TopologyTypeEnum = REQUIRED
    memberships: Mapping[str, str] = field(default_factory=dict)
    edge_ends: tuple[str, str] | None = None
    visualization_type: VisualizationTypeEnum | None = None

    def __post_init__(self):
        object.__setattr__(self, 'memberships', MappingProxyType(dict(self.memberships)))
        if self.edge_ends is not None:
            object.__setattr__(self, 'edge_ends', tuple(self.edge_ends))
        super().__post_init__()
        if not isinstance(self.topology_type, TopologyTypeEnum):
            raise TypeError(f'Spec {self.name!r}: topology_type must be a TopologyTypeEnum.')
        self._check_edge_ends()

    @property
    def schema(self) -> FrameSchema:
        """The levels of the model table: one index level `index` with role `object`, one unnamed column level.

        The columns are of several kinds, for example properties of the objects and memberships.
        No one name fits them all, so the level is unnamed.

        Examples:
            >>> dataset.flag_index.get_spec('Line.Model').schema
            FrameSchema(index=(Level(name='Line', role='object'),), columns=(Level(name=None, role='dimension'),), dtype=None)
        """
        return FrameSchema(index=(Level(self.index, 'object'),), columns=(Level(None, 'dimension'),))

    def _check_edge_ends(self) -> None:
        if self.edge_ends is None:
            return
        if self.topology_type is not TopologyTypeEnum.Edge:
            raise ValueError(f'Spec {self.name!r}: edge_ends needs the topology type Edge.')
        if len(self.edge_ends) != 2 or any(end not in self.memberships for end in self.edge_ends):
            raise ValueError(f'Spec {self.name!r}: edge_ends must name two memberships.')
        if self.memberships[self.edge_ends[0]] != self.memberships[self.edge_ends[1]]:
            raise ValueError(f'Spec {self.name!r}: both edge_ends must refer to the same model flag.')

    def _class_identity(self) -> tuple:
        return self.index, self.topology_type, tuple(sorted(self.memberships.items())), self.edge_ends


@dataclass(frozen=True, eq=False)
class TimeSeriesFlagSpec(FlagSpec):
    """A time series: one time index level, usually one column per object of `model_flag`.

    The topology type and the column level come from the spec of the model flag.

    Attributes:
        model_flag: Linked model flag. `None`: the columns are not objects,
            for example a system total. The spec then declares `schema`.
            It may be a template with placeholders of the variant
            dimensions: with a variant `area_layer` resolved per dataset,
            `'{area_layer}.Model'` links each flag to the model flag of its
            layer, so the flag takes topology type and column level from
            the layer model.
        unit: Unit of the values. `NaU` and `MissingUnit` are not allowed.
        quantity_type: Set only for an ambiguous unit, and required there.
            Else the unit decides.
        schema: Index and column levels of the result, or a function of the
            variant values that returns them. `None`: the default schema.
        fill_value: Value for objects or rows that one side lacks, in sums and
            comparisons. `None`: no fill.

    Examples:
        >>> from mesqual.flag import Dimension
        >>> from mesqual.units import Units
        >>> NODE_SUPPLY_SPEC = TimeSeriesFlagSpec(
        ...     name='Node.Results.supply_{fuel_type}',
        ...     description='Supply at the node.',
        ...     unit=Units.MW,
        ...     model_flag='Node.Model',
        ...     variants=(Dimension('fuel_type', 'Fuel type.', {'res': 'renewable sources', 'coal': 'coal', 'gas': 'gas'}),),
        ...     fill_value=0,
        ... )
    """

    item_type: ClassVar[ItemTypeEnum] = ItemTypeEnum.TimeSeries

    model_flag: str | None = REQUIRED
    unit: Units.Unit = REQUIRED
    quantity_type: QuantityTypeEnum | None = None
    schema: FrameSchema | Callable[[VariantValues], FrameSchema] | None = None
    fill_value: float | None = None

    def __post_init__(self):
        super().__post_init__()
        self._check_unit()
        if self.model_flag is None and self.schema is None:
            raise ValueError(f'Spec {self.name!r}: a time series spec without model_flag must declare a schema.')
        self._check_model_flag_placeholders()
        if isinstance(self.schema, FrameSchema) and len(self.schema.time_levels) != 1:
            raise ValueError(f'Spec {self.name!r}: the schema needs exactly one index level with role time.')

    def _check_unit(self) -> None:
        if not isinstance(self.unit, Units.Unit) or self.unit in (Units.NaU, Units.MissingUnit):
            raise ValueError(f'Spec {self.name!r}: a time series spec needs a unit, got {self.unit!r}.')
        ambiguous = Units.get_quantity_type_enum(self.unit) is QuantityTypeEnum.AMBIGUOUS
        if ambiguous and self.quantity_type not in (QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE):
            raise ValueError(
                f'Spec {self.name!r}: the unit {self.unit} is ambiguous. Set quantity_type to INTENSIVE or EXTENSIVE.'
            )
        if not ambiguous and self.quantity_type is not None:
            raise ValueError(
                f'Spec {self.name!r}: the unit {self.unit} decides the quantity type. Remove quantity_type.'
            )

    def _class_identity(self) -> tuple:
        return self.model_flag, self.unit, self.quantity_type, self.fill_value


@dataclass(frozen=True, eq=False)
class OtherFlagSpec(FlagSpec):
    """Any other table, for example algorithm results per run.

    Attributes:
        schema: Index and column levels of the result, or a function of the
            variant values that returns them. Required.
        unit: Optional unit of the values.
    """

    item_type: ClassVar[ItemTypeEnum] = ItemTypeEnum.Other

    schema: FrameSchema | Callable[[VariantValues], FrameSchema] = REQUIRED
    unit: Units.Unit | None = None

    def _class_identity(self) -> tuple:
        return (self.unit,)
