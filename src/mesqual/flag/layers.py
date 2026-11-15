"""Area layers and border layers: groups of objects and the borders between the groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mesqual.flag.exceptions import AmbiguousOverrideError


@dataclass(frozen=True)
class AreaLayer:
    """A grouping of the objects of one model flag into areas, for example nodes into countries.

    The area of an object is the value of its membership column to `layer_model`.
    The column name appears only in the memberships of the `base_model` spec.

    Attributes:
        name: Layer name, for example `'Region'`. It fills the layer
            placeholder of layer specs.
        index: Label of the object level, for example `'Region'`: the
            index level of the layer model and the column level of the layer
            time series. All flags of the layer share it.
        base_model: Model flag of the grouped objects, for example `'Node.Model'`.
        layer_model: Model flag of the areas, for example `'Region.Model'`.
            The target of exactly one membership of `base_model`.
        description: What the areas are.

    Examples:
        >>> REGION = AreaLayer(
        ...     name='Region',
        ...     index='Region',
        ...     base_model='Node.Model',
        ...     layer_model='Region.Model',
        ...     description='Regions, each a group of nodes through the node column region.',
        ... )
    """

    name: str
    index: str
    base_model: str
    layer_model: str
    description: str


@dataclass(frozen=True)
class BorderLayer:
    """The borders between the areas of one area layer, with the edges that cross them.

    The ends of an edge come from `edge_ends` of the `edge_model` spec. Both
    ends must refer to the `base_model` of the area layer. A border joins two
    areas; the two areas need not be geographic neighbours.

    Attributes:
        name: Layer name, for example `'RegionBorder'`. It fills the
            border layer placeholder of layer specs.
        index: Label of the object level, for example `'RegionBorder'`:
            the index level of the border model and the column level of the
            border time series.
        area_layer: Name of the area layer, for example `'Region'`.
        edge_model: Model flag of the edges that cross borders, for example `'Line.Model'`.
        description: What the borders are.
    """

    name: str
    index: str
    area_layer: str
    edge_model: str
    description: str


class LayerSet:
    """The area and border layers of a dataset, looked up by name.

    Args:
        area_layers: The area layers.
        border_layers: The border layers.

    Raises:
        AmbiguousOverrideError: Two layers have the same name. They would
            give the same flag names.
        ValueError: A border layer names an area layer that is not in the set.
    """

    def __init__(self, area_layers: Iterable[AreaLayer] = (), border_layers: Iterable[BorderLayer] = ()):
        self._area_layers = tuple(area_layers)
        self._border_layers = tuple(border_layers)
        self._by_name: dict[str, AreaLayer | BorderLayer] = {}
        for layer in self._area_layers + self._border_layers:
            if self._by_name.setdefault(layer.name, layer) is not layer:
                raise AmbiguousOverrideError(f'Two layers have the name {layer.name!r}.')
        for layer in self._border_layers:
            if not isinstance(self._by_name.get(layer.area_layer), AreaLayer):
                raise ValueError(f'Border layer {layer.name!r}: no area layer {layer.area_layer!r}.')

    @classmethod
    def union(cls, layer_sets: Iterable[LayerSet]) -> LayerSet:
        """Return one set with the layers of several sets.

        Raises:
            AmbiguousOverrideError: Two sets have different layers with the
                same name. They would give the same flag names.
            ValueError: A border layer names an area layer that is not in any set.
        """
        by_name: dict[str, AreaLayer | BorderLayer] = {}
        for layer_set in layer_sets:
            for layer in layer_set.area_layers + layer_set.border_layers:
                if by_name.setdefault(layer.name, layer) != layer:
                    raise AmbiguousOverrideError(f'Two different layers have the name {layer.name!r}.')
        layers = by_name.values()
        return cls(
            [layer for layer in layers if isinstance(layer, AreaLayer)],
            [layer for layer in layers if isinstance(layer, BorderLayer)],
        )

    @property
    def area_layers(self) -> tuple[AreaLayer, ...]:
        """The area layers, in registration order."""
        return self._area_layers

    @property
    def border_layers(self) -> tuple[BorderLayer, ...]:
        """The border layers, in registration order."""
        return self._border_layers

    def get(self, name: str) -> AreaLayer | BorderLayer:
        """Return the layer with this name.

        Raises:
            KeyError: No layer has this name.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f'No layer {name!r}. Layers: {list(self._by_name)}.') from None

    def area_layer_of(self, border_layer: BorderLayer) -> AreaLayer:
        """Return the area layer of a border layer."""
        return self._by_name[border_layer.area_layer]

    def __bool__(self) -> bool:
        return bool(self._by_name)

    def __repr__(self) -> str:
        return f'LayerSet({list(self._by_name)})'
