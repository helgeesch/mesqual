"""Aggregations of layer flags: from base objects to areas, and from edges or areas to borders."""

from __future__ import annotations

import dataclasses
import logging
import string
import weakref
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from mesqual.energy_data_handling.area_accounting import (
    AreaBorderGeometryCalculator,
    AreaBorderModelGenerator,
    AreaMembership,
    AreaModelGenerator,
    AreaPriceCalculator,
    AreaSumCalculator,
)
from mesqual.flag.layers import AreaLayer, BorderLayer

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset
    from mesqual.datasets.dataset_config import DatasetConfig
    from mesqual.flag.layer_spec import LayerFlagSpec, LayerVariantValues

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LayerAggregation(ABC):
    """Common base of `AreaAggregation` and `BorderAggregation`. Subclasses are frozen dataclasses.

    An aggregation that fetches flags besides the base flag lists them in
    `input_flags`. The layer flag is available only if these inputs are
    available. It passes the `effective_config` that `aggregate` gets to
    each of these fetches, as an interpreter does, so the config of the call
    reaches the inputs. `fetch` passes it, together with the parameters that
    the spec of the input declares.

    Attributes:
        version: Bump by hand when the logic changes. Part of the version
            identity of each layer flag.
    """

    version: int = 1

    def identity(self) -> tuple[str, dict[str, Any]]:
        """Return the link of the aggregation in the version identity: class name and fields.

        Examples:
            >>> WeightedMean(weight='Node.Results.demand_total').identity()
            ('WeightedMean', {'weight': 'Node.Results.demand_total', 'version': 2})
        """
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != 'version'}
        return type(self).__qualname__, {**fields, 'version': self.version}

    def input_flags(
        self,
        dataset: Dataset,
        layer: AreaLayer | BorderLayer,
        variant: LayerVariantValues,
    ) -> tuple[str, ...]:
        """Return the flags that `aggregate` fetches besides the base flag, for example a weight.

        The layer flag interpreter checks their availability without
        fetching them. Default: none.

        Args:
            dataset: The platform dataset.
            layer: The layer of the flag.
            variant: The variant values of the flag, as `aggregate` gets them.
        """
        return ()

    def check_spec(self, spec: LayerFlagSpec) -> None:
        """Raise `ValueError` if the layer spec does not give what the aggregation needs. Default: no check.

        `LayerFlagSpec` calls it at declaration, so a misdeclared spec
        fails there and not in `describe_flags` or `fetch`.
        """

    @staticmethod
    def _variant_names(spec: LayerFlagSpec) -> set[str]:
        """The names in the `variant` that `aggregate` gets: the variants of the spec and the layer of a layer base."""
        names = {d.name for d in spec.variants}
        placeholder = spec._base_layer_placeholder
        return names if placeholder is None else names | {placeholder}

    @staticmethod
    def fetch(
        dataset: Dataset,
        flag: str,
        effective_config: DatasetConfig | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fetch a flag with `effective_config` and the parameters of `params`.

        Parameters that the spec of `flag` does not declare are left out.
        """
        declared = {d.name for d in dataset.flag_index.get_spec(flag).params}
        kwargs = {name: value for name, value in params.items() if name in declared}
        return dataset.fetch(flag, effective_config, **kwargs)


@dataclass(frozen=True, kw_only=True)
class AreaAggregation(LayerAggregation):
    """Aggregates the values of base objects to areas. Subclasses are frozen dataclasses.

    A platform package or a study subclasses it for special cases.
    """

    @abstractmethod
    def aggregate(
        self,
        values: pd.DataFrame | None,
        mapping: pd.Series | pd.DataFrame,
        dataset: Dataset,
        layer: AreaLayer,
        variant: LayerVariantValues,
        effective_config: DatasetConfig,
        **params: Any,
    ) -> pd.DataFrame:
        """Return the aggregated result with the area as object level.

        For a model flag, one row per area. For a time series, one column per
        area; an aggregation on a border base may add column levels, for
        example a partner area.

        Args:
            values: The base flag for the same variant values. On a node
                base, one column per base object, or one row per base object
                for a model flag; objects without area are left out. On a
                border base, one column per border. `None` if the spec
                consumes base variant dimensions, except for a model spec on
                a node base; the aggregation then fetches the variants it
                needs with
                `self.fetch(dataset, variant.base_flag(...), effective_config, params)`.
            mapping: For a node base: base object id to area id, from the
                layer membership column, without objects that have no area.
                For a border base: the border model, with `area_from` and
                `area_to` per border.
            dataset: The platform dataset, to fetch more flags, for example
                weights.
            layer: The layer to aggregate to.
            variant: The variant values of the flag, including the layer. For
                a border base, also the `border_layer` that the interpreter
                resolved for the area layer.
            effective_config: The effective config of the layer flag fetch.
                Pass it to each fetch.
            **params: The resolved parameters of the flag.
        """


@dataclass(frozen=True, kw_only=True)
class BorderAggregation(LayerAggregation):
    """Aggregates edge values or area values to borders. Subclasses are frozen dataclasses."""

    @abstractmethod
    def aggregate(
        self,
        values: pd.DataFrame | None,
        border_model: pd.DataFrame | None,
        dataset: Dataset,
        layer: BorderLayer,
        variant: LayerVariantValues,
        effective_config: DatasetConfig,
        **params: Any,
    ) -> pd.DataFrame:
        """Return one column per border, or one row per border for a model flag.

        Args:
            values: The base flag for the same variant values: one column per
                edge, or per area for an area base. For an edge model base,
                the edge model. `None` if the spec consumes base variant
                dimensions. An aggregation that needs other variant values
                (for example the swapped direction) fetches them with
                `self.fetch(dataset, variant.base_flag(...), effective_config, params)`.
            border_model: The border model of the layer, with `area_from`
                and `area_to` per border. `None` for the border model flag
                itself.
            dataset: The platform dataset, to fetch more flags.
            layer: The layer to aggregate to.
            variant: The variant values of the flag, including the layer. For
                an area base, also the `area_layer` of the border layer.
            effective_config: The effective config of the layer flag fetch.
                Pass it to each fetch.
            **params: The resolved parameters of the flag.
        """


def _require_values(aggregation: LayerAggregation, values: pd.DataFrame | None) -> pd.DataFrame:
    if values is None:
        raise ValueError(f'{type(aggregation).__name__} needs the base values. Do not consume base dimensions.')
    return values


@dataclass(frozen=True, kw_only=True)
class Sum(AreaAggregation):
    """Sum over the objects of each area, with the NaN and index rules of `AreaSumCalculator.calculate`."""

    version: int = 2

    def aggregate(self, values, mapping, dataset, layer, variant, effective_config, **params) -> pd.DataFrame:
        """Return one column per area with the sum of its objects."""
        values = _require_values(self, values)
        return AreaSumCalculator(mapping.to_frame(layer.index), layer.index).calculate(values)


@dataclass(frozen=True, kw_only=True)
class WeightedMean(AreaAggregation):
    """Mean over the objects of each area, weighted by the flag `weight`. Uses `AreaPriceCalculator`.

    Attributes:
        weight: Weight flag. It may use the placeholders of the base name,
            for example `'Node.Results.demand_{product}'`. A time step where
            the weights of an area sum to 0 gives NaN for that area.

    Examples:
        >>> WeightedMean(weight='Node.Results.demand_total')
        WeightedMean(version=2, weight='Node.Results.demand_total')
    """

    weight: str
    version: int = 2

    def check_spec(self, spec: LayerFlagSpec) -> None:
        """Raise `ValueError` if `weight` has a placeholder that is not in the variant values of the spec."""
        fields = {name for _, name, _, _ in string.Formatter().parse(self.weight) if name is not None}
        unknown = fields - self._variant_names(spec)
        if unknown:
            raise ValueError(
                f'Spec {spec.name!r}: the weight {self.weight!r} has placeholders {sorted(unknown)} '
                f'that are no variants of the spec.'
            )

    def input_flags(
        self,
        dataset: Dataset,
        layer: AreaLayer | BorderLayer,
        variant: LayerVariantValues,
    ) -> tuple[str, ...]:
        """Return the weight flag for the variant values."""
        return (self._weight_flag(variant),)

    def aggregate(self, values, mapping, dataset, layer, variant, effective_config, **params) -> pd.DataFrame:
        """Return one column per area with the weighted mean of its objects."""
        values = _require_values(self, values)
        weights = self.fetch(dataset, self._weight_flag(variant), effective_config, params)
        weights = weights.reindex(index=values.index, columns=values.columns)
        return AreaPriceCalculator(mapping.to_frame(layer.index), layer.index).calculate(values, weights)

    def _weight_flag(self, variant: LayerVariantValues) -> str:
        return self.weight.format(**variant.as_dict())


@dataclass(frozen=True, kw_only=True)
class AreaModel(AreaAggregation):
    """The area model: one row per area, with node count and representative point. Uses `AreaModelGenerator`."""

    def aggregate(self, values, mapping, dataset, layer, variant, effective_config, **params) -> pd.DataFrame:
        """Return one row per area, indexed by `layer.index`."""
        objects = _require_values(self, values).loc[mapping.index].assign(**{layer.index: mapping})
        area_model = AreaModelGenerator(objects, layer.index).generate_area_model()
        return area_model.rename_axis(layer.index)


@dataclass(frozen=True, kw_only=True)
class BorderModel(BorderAggregation):
    """The border model: one row per border and direction, with its areas, edge counts and geometry.

    A border runs from `area_from` to `area_to`. Both directions of an area
    pair are borders. An edge counts with the border if its first end is in
    `area_from` and its second end in `area_to`, and against the border the
    other way round. An edge with both ends in one area, or with an end
    without area (`AreaMembership.has_area`), is on no border. So a border
    joins two areas that are both set. The columns `n_edges_with_border` and
    `n_edges_against_border` count the edges per border.

    The columns `opposite_border`, `canonical_border` and `is_canonical` come
    from `AreaBorderModelGenerator`. `canonical_border` is the border of the
    pair with the area names in alphabetical order. `is_canonical` is True
    where the border is its canonical border.

    The model always has the geometry columns (`projection_point`,
    `azimuth_angle`, `geo_line_string`, `areas_touch`). A missing geometry is
    NaN. The values come from `AreaBorderGeometryCalculator`. They need
    polygons in the column `geometry` of the area layer model. The
    calculator gets the CRS of these polygons: the CRS of a `GeoDataFrame`
    or of a geometry column. It reads polygons without a CRS as lon/lat
    coordinates in EPSG:4326. A border whose geometry fails gets NaN in the
    geometry columns, and the fetch logs a warning. If
    the dataset cannot give these polygons, the geometry columns are NaN
    and the fetch logs the reason at `INFO`, once per dataset and area layer.
    If the availability of the area layer model is unknown, the fetch fetches it.

    Attributes:
        geometry: Compute the geometry. With `False`, the geometry columns are NaN.
    """

    N_EDGES_WITH = 'n_edges_with_border'
    N_EDGES_AGAINST = 'n_edges_against_border'
    _LOGGED_WITHOUT_GEOMETRY: ClassVar[weakref.WeakKeyDictionary[Dataset, set[str]]] = weakref.WeakKeyDictionary()
    """Per dataset: the area layers whose border model without geometry is logged at `INFO`."""

    geometry: bool = True
    version: int = 3

    def input_flags(
        self,
        dataset: Dataset,
        layer: AreaLayer | BorderLayer,
        variant: LayerVariantValues,
    ) -> tuple[str, ...]:
        """Return the base model of the area layer. The optional geometry adds no input."""
        return (dataset.flag_index.layers.area_layer_of(layer).base_model,)

    def aggregate(self, values, border_model, dataset, layer, variant, effective_config, **params) -> pd.DataFrame:
        """Return one row per border, indexed by `layer.index`."""
        edges = _require_values(self, values)
        index = dataset.flag_index
        area_layer = index.layers.area_layer_of(layer)
        column = index.get_mapping_column(area_layer)
        nodes = self.fetch(dataset, area_layer.base_model, effective_config, params)
        end_from, end_to = index.get_edge_ends(layer)
        generator = AreaBorderModelGenerator(nodes, edges, column, end_from, end_to, border_identifier=layer.index)
        model = generator.generate_area_border_model()
        model = self._add_edge_counts(model, generator.line_areas)
        if self.geometry:
            model = self._add_geometry(model, generator, dataset, area_layer, effective_config)
        columns = list(dict.fromkeys([*model.columns, *AreaBorderGeometryCalculator.GEOMETRY_COLUMNS]))
        return model.reindex(columns=columns)

    def _add_edge_counts(self, model: pd.DataFrame, edge_areas: pd.DataFrame) -> pd.DataFrame:
        counts = Counter(zip(edge_areas[AreaMembership.AREA_FROM], edge_areas[AreaMembership.AREA_TO]))
        pairs = list(zip(model[AreaMembership.AREA_FROM], model[AreaMembership.AREA_TO]))
        return model.assign(
            **{
                self.N_EDGES_WITH: [counts[(area_from, area_to)] for area_from, area_to in pairs],
                self.N_EDGES_AGAINST: [counts[(area_to, area_from)] for area_from, area_to in pairs],
            }
        )

    def _add_geometry(
        self,
        model: pd.DataFrame,
        generator: AreaBorderModelGenerator,
        dataset: Dataset,
        area_layer: AreaLayer,
        effective_config: DatasetConfig,
    ) -> pd.DataFrame:
        if model.empty:
            return model
        layer_model = area_layer.layer_model
        if not dataset.flag_is_accepted(layer_model):
            return self._without_geometry(model, dataset, area_layer, f'the dataset does not accept {layer_model}')
        if dataset.is_flag_available(layer_model) is False:
            return self._without_geometry(model, dataset, area_layer, f'{layer_model} is unavailable')
        areas = dataset.fetch(layer_model, effective_config)
        if 'geometry' not in areas or not all(isinstance(g, BaseGeometry) for g in areas['geometry']):
            reason = f'{layer_model} has no polygons in the column geometry'
            return self._without_geometry(model, dataset, area_layer, reason)
        calculator = AreaBorderGeometryCalculator(gpd.GeoDataFrame(areas, geometry='geometry'))
        calculator.non_crossing_path_finder.show_progress = False
        return generator.enhance_with_geometry(model, calculator)

    @classmethod
    def _without_geometry(
        cls,
        model: pd.DataFrame,
        dataset: Dataset,
        area_layer: AreaLayer,
        reason: str,
    ) -> pd.DataFrame:
        """Return `model`. Log the reason at `INFO` once per dataset and area layer, then at `DEBUG`."""
        logged = cls._LOGGED_WITHOUT_GEOMETRY.setdefault(dataset, set())
        level = logging.DEBUG if area_layer.name in logged else logging.INFO
        logged.add(area_layer.name)
        logger.log(level, 'Border model of area layer %r without geometry: %s.', area_layer.name, reason)
        return model


@dataclass(frozen=True, kw_only=True)
class BorderEdgeSum(BorderAggregation):
    """Sum of an edge value over the edges of each border, in both directions.

    For edge values without a direction, for example a cost or an income per
    line. An edge from A to B and an edge from B to A both add their value
    to the border A - B and to the border B - A. So both borders of a pair
    have the same value, and a sum over all borders counts each edge twice.
    To count each edge once, select one direction with the column
    `is_canonical` of the border model.

    An edge with both ends in one area, or with an end without area, is on
    no border. A time step where all edges of a border are NaN gives NaN.
    The base needs no variant. It groups the edges by their areas
    (`layer_edge_areas`).
    """

    def input_flags(
        self,
        dataset: Dataset,
        layer: AreaLayer | BorderLayer,
        variant: LayerVariantValues,
    ) -> tuple[str, ...]:
        """Return the inputs of `layer_edge_areas`."""
        return layer_edge_area_inputs(dataset, layer)

    def aggregate(self, values, border_model, dataset, layer, variant, effective_config, **params) -> pd.DataFrame:
        """Return one column per border: its edges in border direction plus its edges against it."""
        values = _require_values(self, values)
        against = self._against_border_values(values, dataset, variant, effective_config, params)
        edge_areas = layer_edge_areas(dataset, layer, effective_config, params)
        return self.sum_by_border(values, edge_areas, border_model, against).rename_axis(columns=layer.index)

    def _against_border_values(self, values, dataset, variant, effective_config, params) -> pd.DataFrame:
        """The values of the edges when they run against the border direction: `values`, without direction."""
        return values

    @classmethod
    def sum_by_border(
        cls,
        values: pd.DataFrame,
        edge_areas: pd.DataFrame,
        border_model: pd.DataFrame,
        against: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return one column per border: the edges in border direction plus the edges against it.

        Args:
            values: One column per edge. An edge in border direction adds
                its value of `values`.
            edge_areas: Two columns per edge, as `AreaMembership.edge_areas`
                or `AreaBorderModelGenerator.line_areas` give them: the first
                with the area of the from end, the second with the area of
                the to end.
            border_model: The border model, with columns of the same two
                names: the source and the target area of each border.
            against: One column per edge. An edge against the border
                direction adds its value of `against`. Default: `values`.

        Returns:
            One column per border of `border_model`, on the index of
                `values`. NaN where all edges of a border are NaN, and for a
                border without edges in `values`.
        """
        against = values if against is None else against
        column_from, column_to = edge_areas.columns
        border_of_pair = dict(zip(zip(border_model[column_from], border_model[column_to]), border_model.index))
        area_from, area_to = edge_areas[column_from], edge_areas[column_to]
        with_border = cls._sum_in_direction(values, area_from, area_to, border_of_pair)
        against_border = cls._sum_in_direction(against, area_to, area_from, border_of_pair)
        return with_border.add(against_border, fill_value=0).reindex(columns=border_model.index)

    @staticmethod
    def _sum_in_direction(
        values: pd.DataFrame,
        area_from: pd.Series,
        area_to: pd.Series,
        border_of_pair: dict[tuple, Any],
    ) -> pd.DataFrame:
        """Sum the edges whose areas run from `area_from` to `area_to` of a border. One column per border."""
        border_of_edge = pd.Series(
            [border_of_pair.get(pair) for pair in zip(area_from, area_to)],
            index=area_from.index,
        )
        border_of_edge = border_of_edge.dropna()
        edges = values.columns.intersection(border_of_edge.index, sort=False)
        return values[edges].T.groupby(border_of_edge[edges]).sum(min_count=1).T


@dataclass(frozen=True, kw_only=True)
class BorderFlowSum(BorderEdgeSum):
    """Sum of the line flows across each border, in border direction.

    A line against the border direction counts with its directions swapped:
    its `down` adds to the border `up`, its `up` to the border `down`, and
    its `net` with a minus sign. `total` (up plus down) is the same in both
    directions. The base needs a `direction` variant with the values `up`,
    `down` and, if used, `net` and `total`. Any other value raises
    `ValueError`. For `up` and `down` it fetches the swapped variant through
    `self.fetch`. It groups the lines by their areas (`layer_edge_areas`).

    For example, a line from B to A on the border A-B with up 30 and down 50
    adds 50 to the border up and 30 to the border down.
    """

    DIRECTION = 'direction'

    def check_spec(self, spec: LayerFlagSpec) -> None:
        """Raise `ValueError` if the spec has no `direction` variant."""
        if self.DIRECTION not in self._variant_names(spec):
            raise ValueError(f'Spec {spec.name!r}: {type(self).__name__} needs a {self.DIRECTION!r} variant.')

    def input_flags(
        self,
        dataset: Dataset,
        layer: AreaLayer | BorderLayer,
        variant: LayerVariantValues,
    ) -> tuple[str, ...]:
        """Return the inputs of `layer_edge_areas`, and the swapped base variant for `up` and `down`."""
        swapped = self._swapped_direction(getattr(variant, self.DIRECTION))
        swapped_flag = () if swapped is None else (variant.base_flag(**{self.DIRECTION: swapped}),)
        return layer_edge_area_inputs(dataset, layer) + swapped_flag

    def _against_border_values(self, values, dataset, variant, effective_config, params) -> pd.DataFrame:
        """The values of the lines against the border direction: the swapped direction, or `-net`, or `total`."""
        direction = getattr(variant, self.DIRECTION)
        swapped = self._swapped_direction(direction)
        if swapped is not None:
            return self.fetch(dataset, variant.base_flag(**{self.DIRECTION: swapped}), effective_config, params)
        if direction == 'net':
            return -values
        if direction == 'total':
            return values
        raise ValueError(f'{type(self).__name__}: unknown direction {direction!r}. Use up, down, net or total.')

    @staticmethod
    def _swapped_direction(direction: str) -> str | None:
        """`down` for `up` and `up` for `down`, else `None`."""
        return {'up': 'down', 'down': 'up'}.get(direction)


def layer_edge_area_inputs(dataset: Dataset, layer: BorderLayer) -> tuple[str, str]:
    """The flags that `layer_edge_areas` fetches: the base model of the area layer and the edge model of `layer`."""
    return dataset.flag_index.layers.area_layer_of(layer).base_model, layer.edge_model


def layer_edge_areas(
    dataset: Dataset,
    layer: BorderLayer,
    effective_config: DatasetConfig | None,
    params: dict[str, Any],
) -> pd.DataFrame:
    """`AreaMembership.edge_areas` for the edge model of `layer`, with the areas of its area layer.

    It fetches the flags of `layer_edge_area_inputs` with `LayerAggregation.fetch`.

    Args:
        effective_config: The config for the fetches: the effective config
            of the enclosing fetch, or `None` for the config of `dataset`
            (outside a fetch).
    """
    index = dataset.flag_index
    area_layer = index.layers.area_layer_of(layer)
    node_model, edge_model = layer_edge_area_inputs(dataset, layer)
    nodes = LayerAggregation.fetch(dataset, node_model, effective_config, params)
    edges = LayerAggregation.fetch(dataset, edge_model, effective_config, params)
    return AreaMembership.edge_areas(edges, nodes[index.get_mapping_column(area_layer)], *index.get_edge_ends(layer))


@dataclass(frozen=True, kw_only=True)
class BorderCapacitySum(BorderFlowSum):
    """Same as `BorderFlowSum`, on line capacities."""


@dataclass(frozen=True, kw_only=True)
class PriceSpread(BorderAggregation):
    """Area value at the end of the border minus the value at its start. Base: an area spec."""

    def aggregate(self, values, border_model, dataset, layer, variant, effective_config, **params) -> pd.DataFrame:
        """Return one column per border: value of `area_to` minus value of `area_from`."""
        values = _require_values(self, values)
        missing = pd.Series(float('nan'), index=values.index)
        result = {
            border: values.get(row[AreaMembership.AREA_TO], missing)
            - values.get(row[AreaMembership.AREA_FROM], missing)
            for border, row in border_model.iterrows()
        }
        return pd.DataFrame(result, index=values.index).rename_axis(columns=layer.index)
